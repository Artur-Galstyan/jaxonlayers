import hashlib
import os

import clu.metrics as clum
import equinox as eqx
import flax
import grain
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mlflow
import numpy as np
import optax
import polars as pl
import tensorflow_datasets as tfds
from tqdm import tqdm

PERCENTILE = 90
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = ["Z"] + list(AMINO_ACIDS)
INT_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}
SPLITS = {
    "train": "solubility_training.csv",
    "validation": "solubility_validation.csv",
    "test": "solubility_testing.csv",
}
TRAIN_DF = pl.read_csv("hf://datasets/proteinea/solubility/" + SPLITS["train"])
TEST_DF = pl.read_csv("hf://datasets/proteinea/solubility/" + SPLITS["test"])
VALIDATION_DF = pl.read_csv(
    "hf://datasets/proteinea/solubility/" + SPLITS["validation"]
)

TRAIN_DF = TRAIN_DF.with_columns(
    pl.col("sequences").str.len_chars().alias("length"),
)
MAX_PROTEIN_LENGTH = int(
    TRAIN_DF.select(pl.col("length").quantile(PERCENTILE / 100)).item()
)


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")  # pyright: ignore
    accuracy: clum.Average.from_output("accuracy")  # pyright: ignore


class ProteinPreprocessor(grain.transforms.Map):
    def map(self, example):
        sequence = example["sequences"]
        label = example["labels"]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        sequence = sequence[:MAX_PROTEIN_LENGTH]
        sequence = sequence.ljust(MAX_PROTEIN_LENGTH, "Z")
        indices = [CHAR_TO_INT[aa] for aa in sequence]
        return {
            "features": jnp.array(indices, jnp.int32),
            "sequence": sequence,
            "label": jnp.zeros(shape=(2,)).at[label].set(1),
        }


class SimpleModel(eqx.Module):
    mlp: eqx.nn.MLP
    dropout: eqx.nn.Dropout
    linear: eqx.nn.Linear

    def __init__(self, embedding_size: int, out_features: int, key: jt.PRNGKeyArray):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        hidden_size = 256
        self.mlp = eqx.nn.MLP(
            embedding_size, hidden_size, width_size=hidden_size, depth=4, key=k1
        )
        self.dropout = eqx.nn.Dropout()
        self.linear = eqx.nn.Linear(hidden_size, out_features, key=k2)

    def __call__(
        self,
        x: jt.Float[jt.Array, " embedding_size"],
        key: jt.PRNGKeyArray | None,
        inference: bool,
    ) -> jt.Array:
        x = self.mlp(x)
        x = self.dropout(x, inference=inference, key=key)
        x = self.linear(x)
        return x


def load_embeddings(data: dict):
    xs = []
    for seq in data["sequence"]:
        seq_sha = hashlib.sha256(seq.encode("utf-8")).hexdigest()
        if not os.path.exists(f"experiments/embeddings/{seq_sha}.npy"):
            raise FileNotFoundError(
                f"experiments/embeddings/{seq_sha}.npy does not exist"
            )
        embedding = np.load(f"experiments/embeddings/{seq_sha}.npy")
        pooled_embedding = np.mean(embedding, axis=1)
        xs.append(pooled_embedding)
    x = jnp.array(xs).reshape(len(xs), -1)
    return x


def loss_fn(
    model: SimpleModel,
    x: jt.Array,
    y: jt.Array,
    key: jt.PRNGKeyArray | None,
    inference: bool,
) -> tuple[jt.Array, jt.Array]:
    if inference is False and key is not None:
        keys = jax.random.split(key, x.shape[0])
        in_axes = (0, 0, None)
    else:
        keys = key
        in_axes = (0, None, None)

    logits = eqx.filter_vmap(model, in_axes=in_axes)(x, keys, inference)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y)), logits


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jt.PRNGKeyArray,
) -> tuple[SimpleModel, optax.OptState, dict]:
    print("step_fn JIT")
    inference = False
    (loss_value, preds), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y, key, inference
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
    key, inference = None, True
    logits = eqx.filter_vmap(model, in_axes=(0, None, None))(x, key, inference)
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
        x = load_embeddings(data)
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


def train(
    n_epochs: int,
    steps_per_epoch: int,
    dataset_size: int,
    train_source,
    test_source,
    batch_size: int,
    model: SimpleModel,
    learning_rate: float,
):
    weight_decay = 1e-4
    run_description = "Using embeddings from ESM-C to establish a baseline; small mlp; using weight decay with adamw"
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_metrics = []
    eval_metrics = []

    key = jax.random.key(12)
    with mlflow.start_run(description=run_description):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "model": str(model),
                "weight_decay": weight_decay,
            }
        )
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
                x = load_embeddings(data)
                y = data["label"]
                key, step_key = jax.random.split(key)
                model, opt_state, step_metrics = step_fn(
                    model, x, y, optimizer, opt_state, step_key
                )
                epoch_train_metrics = epoch_train_metrics.merge(
                    LossMetrics.single_from_model_output(
                        loss=step_metrics["loss"], accuracy=step_metrics["accuracy"]
                    )
                )
                epoch_steps += 1

                computed_metrics = epoch_train_metrics.compute()
                pbar.set_postfix(
                    loss_and_acc=f"Loss: {computed_metrics['loss']:.4f}, Accuracy: {computed_metrics['accuracy']:.4f}"
                )

                if epoch_steps >= steps_per_epoch:
                    break

            test_dataset_size = len(test_source)
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

            epoch_eval_metrics = eval_fn(model, test_data_loader)
            mlflow.log_metrics(
                {
                    "train_loss": float(epoch_train_metrics.compute()["loss"]),
                    "train_accuracy": float(epoch_train_metrics.compute()["accuracy"]),
                    "test_loss": float(epoch_eval_metrics["loss"]),
                    "test_accuracy": float(epoch_eval_metrics["accuracy"]),
                },
                step=epoch,
            )
            train_metrics.append(epoch_train_metrics)
            eval_metrics.append(epoch_eval_metrics)

    return model, opt_state, train_metrics, eval_metrics


def main():
    experiment_name = "protein_solubility_classification"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.enable_system_metrics_logging()

    out_features = 2
    embedding_size = 960  # this is from esm-c
    learning_rate = 3e-4
    n_epochs = 15
    batch_size = 128

    builder = tfds.dataset_builders.CroissantBuilder(
        jsonld="https://huggingface.co/api/datasets/proteinea/solubility/croissant",
        file_format="array_record",
    )
    builder.download_and_prepare()

    train_source, test_source = builder.as_data_source(split=["train", "test"])

    dataset_size = len(train_source)
    steps_per_epoch = dataset_size // batch_size
    print(f"Dataset size: {dataset_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

    model = SimpleModel(embedding_size, out_features, key=jax.random.key(42))

    model, opt_state, train_metrics, eval_metrics = train(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        dataset_size=dataset_size,
        train_source=train_source,
        test_source=test_source,
        batch_size=batch_size,
        model=model,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    main()
