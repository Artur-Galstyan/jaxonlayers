import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def sinusoidal_embedding(
    t: Int[Array, ""], embedding_size: int
) -> Float[Array, " embedding_size"]:
    if embedding_size % 2 != 0:
        raise ValueError(f"Embedding size must be even, but got {embedding_size}")

    half_dim = embedding_size // 2
    embedding_freqs = jnp.exp(
        -jnp.log(10000)
        * jnp.arange(start=0, stop=half_dim, dtype=jnp.float32)
        / half_dim
    )

    time_args = t * embedding_freqs
    embedding = jnp.concatenate([jnp.sin(time_args), jnp.cos(time_args)], axis=-1)

    return embedding
