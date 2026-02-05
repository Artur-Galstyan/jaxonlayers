import jax
import jax.numpy as jnp
from jaxtyping import Array


def swiglu(x: Array, axis=-1):
    a, b = jnp.split(x, 2, axis=axis)
    return a * jax.nn.swish(b)
