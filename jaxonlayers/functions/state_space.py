import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def selective_scan(
    u: Float[Array, "seq_length d_inner"],
    delta: Float[Array, "seq_length d_inner"],
    A: Float[Array, "d_inner d_state"],
    B: Float[Array, "seq_length d_inner d_state"],
    C: Float[Array, "seq_length d_inner d_state"],
    D: Float[Array, "d_inner"],
) -> Float[Array, "seq_length d_inner"]:
    L, d_inner = u.shape
    _, d_state = A.shape

    deltaA = jnp.exp(jnp.einsum("l d, d n -> l d n", delta, A))
    deltaB_u = jnp.einsum("l d, l d n, l d -> l d n", delta, B, u)

    h0 = jnp.zeros((d_inner, d_state))

    def step(h_prev, scan_inputs):
        deltaA_i, deltaB_u_i, C_i = scan_inputs
        h_i = deltaA_i * h_prev + deltaB_u_i
        y_i = jnp.einsum("d n, d n -> d", h_i, C_i).real
        return h_i, y_i

    _, ys = jax.lax.scan(step, h0, (deltaA, deltaB_u, C))

    ys = ys + u * D
    return ys
