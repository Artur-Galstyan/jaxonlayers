import os

import clu.metrics as clum
import equinox as eqx
import flax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mlflow
import optax
from data import VOCAB, get_dataloaders, get_datasources
from tabulate import tabulate
from tqdm import tqdm

from jaxonlayers.layers import LayerNorm, MultiheadAttention

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RUN_DESCRIPTION = "MHA"


class TransformerBlock(eqx.Module):
    mha: MultiheadAttention
    norm1: LayerNorm
    norm2: LayerNorm
    ffn: eqx.nn.MLP

    def __init__(self, embed_dim: int, num_heads: int, key: jt.PRNGKeyArray):
        k1, k2 = jax.random.split(key, 2)
        self.mha = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            inference=True,
            key=k1,
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=embed_dim * 4,
            depth=1,
            key=k2,
        )

    def __call__(
        self, x: jt.Array, state: eqx.nn.State
    ) -> tuple[jt.Array, eqx.nn.State]:
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = x + attn_out
        # x = self.norm1(x)

        ffn_out = eqx.filter_vmap(self.ffn)(x)
        x = x + ffn_out
        # x = self.norm2(x)

        return x, state


class SimpleModel(eqx.Module):
    embedding: eqx.nn.Embedding
    blocks: list
    mlp: eqx.nn.MLP
    linear: eqx.nn.Linear

    def __init__(
        self,
        n_vocab: int,
        seq_len: int,
        embedding_size: int,
        out_features: int,
        key: jt.PRNGKeyArray,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        d_inner = 128
        num_blocks = 3

        self.embedding = eqx.nn.Embedding(n_vocab, embedding_size, key=k2)

        block_keys = jax.random.split(k3, num_blocks)
        self.blocks = [
            TransformerBlock(embedding_size, num_heads=8, key=bk) for bk in block_keys
        ]

        self.mlp = eqx.nn.MLP(
            embedding_size, d_inner, width_size=d_inner, depth=4, key=k1
        )
        self.linear = eqx.nn.Linear(d_inner, out_features, key=k4)

    def __call__(
        self, x: jt.Float[jt.Array, " seq_len"], state: eqx.nn.State, key, inference
    ) -> tuple[jt.Array, eqx.nn.State]:
        x = eqx.filter_vmap(self.embedding)(x)

        for block in self.blocks:
            x, state = block(x, state)

        x = jnp.mean(x, axis=0)
        x = self.mlp(x)
        x = self.linear(x)
        return x, state


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")
    accuracy: clum.Average.from_output("accuracy")


def loss_fn(
    model: SimpleModel,
    state: eqx.nn.State,
    x: jt.Array,
    y: jt.Array,
    key: jt.PRNGKeyArray | None,
    inference: bool,
) -> tuple[jt.Array, tuple[jt.Array, eqx.nn.State]]:
    if inference is False and key is not None:
        keys = jax.random.split(key, x.shape[0])
        in_axes = (0, None, 0, None)
    else:
        keys = key
        in_axes = (0, None, None, None)

    logits, state = eqx.filter_vmap(
        model, in_axes=in_axes, out_axes=(0, None), axis_name="batch"
    )(x, state, keys, inference)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y)), (
        logits,
        state,
    )


@eqx.filter_jit
def step_fn(
    model: SimpleModel,
    state: eqx.nn.State,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jt.PRNGKeyArray,
) -> tuple[SimpleModel, eqx.nn.State, optax.OptState, dict]:
    print("step_fn JIT")
    inference = False
    (loss_value, (preds, state)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, state, x, y, key, inference)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    predicted_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    accuracy = jnp.mean(predicted_classes == true_classes)
    metrics = {"loss": loss_value, "accuracy": accuracy}
    return model, state, opt_state, metrics


@eqx.filter_jit
def eval_step(model: SimpleModel, state: eqx.nn.State, x: jt.Array, y: jt.Array):
    print("eval_step JIT")
    key, inference = None, True
    logits, state = eqx.filter_vmap(
        model, in_axes=(0, None, None, None), out_axes=(0, None), axis_name="batch"
    )(x, state, key, inference)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))
    predicted_classes = jnp.argmax(logits, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)
    correct_preds = jnp.sum(predicted_classes == true_classes)
    return loss, correct_preds


def eval_fn(model: SimpleModel, state: eqx.nn.State, loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for data in loader:
        y = data["label"]
        x = data["features"]
        num_samples = x.shape[0]
        batch_loss, batch_correct = eval_step(model, state, x, y)
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
    train_source,
    test_source,
    validation_source,
    batch_size: int,
    model: SimpleModel,
    state: eqx.nn.State,
    learning_rate: float,
    max_protein_length: int,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.key(12)

    results_table = []

    with mlflow.start_run(description=RUN_DESCRIPTION):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "model": str(model),
                "max_protein_length": max_protein_length,
            }
        )

        for epoch in range(n_epochs):
            train_data_loader, test_data_loader, validation_data_loader = (
                get_dataloaders(
                    train_source=train_source,
                    test_source=test_source,
                    validation_source=validation_source,
                    epoch=epoch,
                    batch_size=batch_size,
                    max_protein_length=max_protein_length,
                )
            )

            epoch_steps = 0
            epoch_train_metrics = LossMetrics.empty()
            pbar = tqdm(
                train_data_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for data in pbar:
                x = data["features"]
                y = data["label"]
                key, step_key = jax.random.split(key)
                model, state, opt_state, step_metrics = step_fn(
                    model, state, x, y, optimizer, opt_state, step_key
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

            epoch_eval_metrics = eval_fn(model, state, test_data_loader)
            epoch_validation_metrics = eval_fn(model, state, validation_data_loader)

            train_metrics = epoch_train_metrics.compute()

            results_table.append(
                [
                    epoch,
                    f"{float(train_metrics['loss']):.4f}",
                    f"{float(train_metrics['accuracy']):.4f}",
                    f"{float(epoch_eval_metrics['loss']):.4f}",
                    f"{float(epoch_eval_metrics['accuracy']):.4f}",
                    f"{float(epoch_validation_metrics['loss']):.4f}",
                    f"{float(epoch_validation_metrics['accuracy']):.4f}",
                ]
            )

            print(
                "\n"
                + tabulate(
                    results_table,
                    headers=[
                        "Epoch",
                        "Train Loss",
                        "Train Acc",
                        "Test Loss",
                        "Test Acc",
                        "Val Loss",
                        "Val Acc",
                    ],
                    tablefmt="simple_grid",
                )
            )

            mlflow.log_metrics(
                {
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "test_loss": float(epoch_eval_metrics["loss"]),
                    "test_accuracy": float(epoch_eval_metrics["accuracy"]),
                    "validation_loss": float(epoch_validation_metrics["loss"]),
                    "validation_accuracy": float(epoch_validation_metrics["accuracy"]),
                },
                step=epoch,
            )

    return model, opt_state


def main():
    print("Experiment:", RUN_DESCRIPTION)
    experiment_name = "protein_solubility_classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.enable_system_metrics_logging()

    out_features = 2
    embedding_size = 256
    learning_rate = 3e-4
    n_epochs = 50
    batch_size = 128
    percentile = 90

    train_source, test_source, validation_source, max_protein_length = get_datasources(
        percentile
    )

    dataset_size = len(train_source)
    steps_per_epoch = dataset_size // batch_size

    model, state = eqx.nn.make_with_state(SimpleModel)(
        len(VOCAB),
        max_protein_length,
        embedding_size,
        out_features,
        key=jax.random.key(42),
    )

    train(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        train_source=train_source,
        test_source=test_source,
        validation_source=validation_source,
        batch_size=batch_size,
        model=model,
        state=state,
        learning_rate=learning_rate,
        max_protein_length=max_protein_length,
    )


if __name__ == "__main__":
    main()
