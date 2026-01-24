import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def default_floating_dtype():
    if jax.config.read("jax_enable_x64"):  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def summarize_model(model: PyTree) -> str:
    params, _ = eqx.partition(model, eqx.is_array)

    param_counts = {}
    total_params = 0

    def count_params(pytree, name=""):
        nonlocal total_params
        count = 0
        if isinstance(pytree, jnp.ndarray):
            count = pytree.size
            total_params += count
            if name:
                param_counts[name] = count
        elif hasattr(pytree, "__dict__"):
            for key, value in pytree.__dict__.items():
                subname = f"{name}.{key}" if name else key
                count += count_params(value, subname)
        elif isinstance(pytree, (list, tuple)):
            for i, value in enumerate(pytree):
                subname = f"{name}[{i}]" if name else f"[{i}]"
                count += count_params(value, subname)
        elif isinstance(pytree, dict):
            for key, value in pytree.items():
                subname = f"{name}.{key}" if name else str(key)
                count += count_params(value, subname)
        return count

    count_params(params)

    # Display as table
    lines = []
    lines.append("Model Parameter Summary")
    lines.append("=" * 50)
    lines.append(f"{'Parameter Name':<30} {'Count':<15}")
    lines.append("-" * 50)

    for name, count in param_counts.items():
        lines.append(f"{name:<30} {count:<15,}")

    lines.append("-" * 50)
    lines.append(f"{'Total Parameters':<30} {total_params:<15,}")
    lines.append("=" * 50)

    return "\n".join(lines)
