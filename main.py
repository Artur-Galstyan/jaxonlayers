import clu.metrics as clum
import equinox as eqx
import flax
import grain
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import polars as pl
from beartype.typing import Any
from jaxonlayers.functions.utils import default_floating_dtype
from jaxonlayers.layers.state_space import SelectiveStateSpace
from tqdm import tqdm

import tensorflow_datasets as tfds

out_features = 2
embedding_size = 960  # this is from esm-c
learning_rate = 0.001
n_epochs = 10


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")  # pyright: ignore
    accuracy: clum.Average.from_output("accuracy")  # pyright: ignore


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vocab = ["Z"] + list(amino_acids)
int_to_char = {i: char for i, char in enumerate(vocab)}
char_to_int = {char: i for i, char in enumerate(vocab)}

splits = {
    "train": "solubility_training.csv",
    "validation": "solubility_validation.csv",
    "test": "solubility_testing.csv",
}
train_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["train"])
test_df = pl.read_csv("hf://datasets/proteinea/solubility/" + splits["test"])
validation_df = pl.read_csv(
    "hf://datasets/proteinea/solubility/" + splits["validation"]
)
PERCENTILE = 90

train_df = train_df.with_columns(
    pl.col("sequences").str.len_chars().alias("length"),
)
max_protein_length = int(
    train_df.select(pl.col("length").quantile(PERCENTILE / 100)).item()
)


class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[:max_protein_length]
        sequence = sequence.ljust(max_protein_length, "Z")
        indices = [char_to_int[aa] for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "sequence": sequence,
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }


class MambaBlock(eqx.Module):
    in_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d
    ssm: SelectiveStateSpace
    out_proj: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    d_model: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        conv_kernel_size: int = 4,
        *,
        key: jt.PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        self.d_model = d_model
        self.d_inner = d_inner
        keys = jax.random.split(key, 4)

        self.in_proj = eqx.nn.Linear(
            d_model, 2 * d_inner, use_bias=False, key=keys[0], dtype=dtype
        )

        self.conv1d = eqx.nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=conv_kernel_size,
            groups=d_inner,
            padding=conv_kernel_size - 1,
            key=keys[1],
            dtype=dtype,
        )

        self.ssm = SelectiveStateSpace(
            d_model=d_inner,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_state,
            key=keys[2],
            dtype=dtype,
        )

        self.out_proj = eqx.nn.Linear(
            d_inner, d_model, use_bias=False, key=keys[3], dtype=dtype
        )

        self.norm = eqx.nn.LayerNorm(d_model, eps=1e-5, dtype=dtype)

    def __call__(self, x: jt.Float[jt.Array, "seq_length d_model"]):
        seq_len, _ = x.shape
        x_norm = eqx.filter_vmap(self.norm)(x)

        x_proj = jax.vmap(self.in_proj)(x_norm)
        main_path, gate = jnp.split(x_proj, 2, axis=-1)

        main_path_transposed = main_path.T
        main_path_conv = self.conv1d(main_path_transposed)
        main_path_conv = main_path_conv[:, :seq_len]
        main_path_conv = main_path_conv.T

        main_path_activated = jax.nn.silu(main_path_conv)

        y = self.ssm(main_path_activated)

        gate_activated = jax.nn.silu(gate)
        output = y * gate_activated

        output = jax.vmap(self.out_proj)(output)

        return x + output


class BidirectionalMambaLayer(eqx.Module):
    forward_block: MambaBlock
    backward_block: MambaBlock

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        conv_kernel_size: int = 4,
        *,
        key: jt.PRNGKeyArray,
        dtype: Any | None = None,
    ):
        fwd_key, bwd_key = jax.random.split(key)

        self.forward_block = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_state,
            conv_kernel_size=conv_kernel_size,
            key=fwd_key,
            dtype=dtype,
        )
        self.backward_block = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_state,
            conv_kernel_size=conv_kernel_size,
            key=bwd_key,
            dtype=dtype,
        )

    def __call__(self, x: jt.Float[jt.Array, "seq_length d_model"], *, key):
        fwd_out = self.forward_block(x)

        x_rev = jnp.flip(x, axis=0)
        bwd_out_rev = self.backward_block(x_rev)
        bwd_out = jnp.flip(bwd_out_rev, axis=0)

        return fwd_out + bwd_out


class SSMEncoder(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: eqx.nn.Sequential
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        conv_kernel_size: int = 4,
        *,
        key: jt.PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        keys = jax.random.split(key, num_layers + 1)

        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=d_model, key=keys[0]
        )

        layer_list = []
        for i in range(num_layers):
            layer = BidirectionalMambaLayer(
                d_model=d_model,
                d_inner=d_inner,
                dt_rank=dt_rank,
                d_state=d_state,
                conv_kernel_size=conv_kernel_size,
                key=keys[i + 1],
                dtype=dtype,
            )
            layer_list.append(layer)

        self.layers = eqx.nn.Sequential(layer_list)
        self.norm = eqx.nn.LayerNorm(d_model, eps=1e-5, dtype=dtype)

    def __call__(self, x: jt.Int[jt.Array, "seq_length"]):
        x_embed = jax.vmap(self.embedding)(x)
        x_processed = self.layers(x_embed)
        x_norm = eqx.filter_vmap(self.norm)(x_processed)
        return x_norm


class SimpleModel(eqx.Module):
    mlp: eqx.nn.MLP
    # embedding: eqx.nn.Embedding
    encoder: SSMEncoder

    def __init__(
        self, n_vocab: int, embedding_size: int, out_features: int, key: jt.PRNGKeyArray
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        num_layers = 3
        d_model = 64
        d_inner = 128
        d_state = 8
        dt_rank = 8
        self.encoder = SSMEncoder(
            num_layers=num_layers,
            vocab_size=n_vocab,
            d_model=d_model,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_state,
            key=k3,
        )

        # self.embedding = eqx.nn.Embedding(n_vocab, embedding_size, key=k2)
        self.mlp = eqx.nn.MLP(
            d_model, out_features, width_size=d_model, depth=3, key=k1
        )

    def __call__(self, x: jt.Float[jt.Array, " seq_len"]) -> jt.Array:
        # x = eqx.filter_vmap(self.embedding)(x)
        x = self.encoder(x)
        x = jnp.mean(x, axis=0)
        x = self.mlp(x)
        return x


def loss_fn(model: SimpleModel, x: jt.Array, y: jt.Array) -> tuple[jt.Array, jt.Array]:
    logits = eqx.filter_vmap(model)(x)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y)), logits


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[SimpleModel, optax.OptState, dict]:
    print("step_fn JIT")
    (loss_value, preds), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    predicted_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    accuracy = jnp.mean(predicted_classes == true_classes)

    metrics = {"loss": loss_value, "accuracy": accuracy}
    return model, opt_state, metrics


@eqx.filter_jit
def eval_step(model: SimpleModel, x: jt.Array, y: jt.Array):
    print("eval_step JIT")
    logits = eqx.filter_vmap(model)(x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))

    predicted_classes = jnp.argmax(logits, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    correct_preds = jnp.sum(predicted_classes == true_classes)

    return loss, correct_preds


def eval_fn(model: SimpleModel, loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data in loader:
        y = data["label"]
        x = data["features"].reshape(batch_size, -1)
        num_samples = x.shape[0]
        batch_loss, batch_correct = eval_step(model, x, y)

        total_loss += batch_loss * num_samples
        total_correct += batch_correct
        total_samples += num_samples

    if total_samples == 0:
        return {"loss": float("inf"), "accuracy": 0.0}

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


batch_size = 16
builder = tfds.dataset_builders.CroissantBuilder(
    jsonld="https://huggingface.co/api/datasets/proteinea/solubility/croissant",
    file_format="array_record",
)
builder.download_and_prepare()

train_source, test_source = builder.as_data_source(split=["train[:80%]", "train[80%:]"])

dataset_size = len(train_source)
steps_per_epoch = dataset_size // batch_size
print(f"Dataset size: {dataset_size}")
print(f"Steps per epoch: {steps_per_epoch}")

train_index_sampler = grain.samplers.IndexSampler(
    num_records=dataset_size,
    shuffle=True,
    shard_options=grain.sharding.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True
    ),
    seed=4,
    num_epochs=1,
)

train_data_loader = grain.DataLoader(
    data_source=train_source,  # pyright: ignore
    operations=[ProteinPreprocessor(), grain.transforms.Batch(batch_size=batch_size)],
    sampler=train_index_sampler,
)

train_metrics = LossMetrics.empty()


in_features = len(vocab)
model = SimpleModel(in_features, embedding_size, out_features, key=jax.random.key(42))
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


train_metrics = LossMetrics.empty()
test_metrics = LossMetrics.empty()


for epoch in range(n_epochs):
    train_index_sampler = grain.samplers.IndexSampler(
        num_records=dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=4 + epoch,
        num_epochs=1,
    )

    train_data_loader = grain.DataLoader(
        data_source=train_source,  # pyright: ignore
        operations=[
            ProteinPreprocessor(),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=train_index_sampler,
    )

    epoch_steps = 0
    epoch_train_metrics = LossMetrics.empty()
    pbar = tqdm(train_data_loader, total=steps_per_epoch, desc=f"Epoch {epoch}")
    for data in pbar:
        x = data["features"].reshape(batch_size, -1)
        y = data["label"]
        model, opt_state, step_metrics = step_fn(model, x, y, optimizer, opt_state)
        epoch_train_metrics = epoch_train_metrics.merge(
            LossMetrics.single_from_model_output(
                loss=step_metrics["loss"], accuracy=step_metrics["accuracy"]
            )
        )
        epoch_steps += 1

        computed = epoch_train_metrics.compute()
        train_loss_tqdm = computed["loss"]
        train_accuracy_tqdm = computed["accuracy"]
        pbar.set_postfix(
            loss_and_acc=f"loss={train_loss_tqdm:.4f}, acc={train_accuracy_tqdm:.4f}"
        )

        if epoch_steps >= steps_per_epoch:
            break

    test_dataset_size = len(test_source)
    test_steps = test_dataset_size // batch_size

    test_index_sampler = grain.samplers.IndexSampler(
        num_records=test_dataset_size,
        shuffle=False,
        shard_options=grain.sharding.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        seed=42,
        num_epochs=1,
    )

    test_data_loader = grain.DataLoader(
        data_source=test_source,  # pyright: ignore
        operations=[
            ProteinPreprocessor(),
            grain.transforms.Batch(batch_size=batch_size),
        ],
        sampler=test_index_sampler,
    )

    print(f"Epoch: {epoch}")
    print(f"Train - {epoch_train_metrics.compute()}")
    eval_metrics = eval_fn(model, test_data_loader)
    print(
        f"Test  - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}"
    )
