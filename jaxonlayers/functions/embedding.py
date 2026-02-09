import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, Int

from jaxonlayers.functions import default_floating_dtype


def sinusoidal_embedding(
    t: Int[Array, ""], embedding_size: int, dtype: Any | None = None
) -> Float[Array, " embedding_size"]:
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    if embedding_size % 2 != 0:
        raise ValueError(f"Embedding size must be even, but got {embedding_size}")

    half_dim = embedding_size // 2
    embedding_freqs = jnp.exp(
        -jnp.log(10000) * jnp.arange(start=0, stop=half_dim, dtype=dtype) / half_dim
    )

    time_args = t * embedding_freqs
    embedding = jnp.concatenate([jnp.sin(time_args), jnp.cos(time_args)], axis=-1)

    return embedding
