import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonlayers.functions import default_floating_dtype, selective_scan


class SelectiveStateSpace(eqx.Module):
    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, "d_inner"]
    out_proj: eqx.nn.Linear

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state

        keys = jax.random.split(key, 4)
        proj_dim = self.dt_rank + 2 * self.d_inner * self.d_state
        self.input_proj = eqx.nn.Linear(
            self.d_model,
            proj_dim,
            use_bias=use_input_proj_bias,
            key=keys[0],
            dtype=dtype,
        )

        self.delta_proj = eqx.nn.Linear(
            dt_rank,
            d_inner,
            use_bias=use_delta_proj_bias,
            key=keys[1],
            dtype=dtype,
        )

        A = jnp.arange(1, d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A, (d_inner, 1))
        self.A_log = jnp.log(A)

        self.D = jnp.ones(d_inner, dtype=dtype)

        self.out_proj = eqx.nn.Linear(
            d_inner, d_model, use_bias=False, key=keys[2], dtype=dtype
        )

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        L, _ = x.shape
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)

        delta_b_c = jax.vmap(self.input_proj)(x)
        delta, B, C = jnp.split(
            delta_b_c,
            [self.dt_rank, self.dt_rank + self.d_inner * self.d_state],
            axis=-1,
        )

        B = B.reshape(L, self.d_inner, self.d_state)
        C = C.reshape(L, self.d_inner, self.d_state)

        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)

        return jax.vmap(self.out_proj)(y)
