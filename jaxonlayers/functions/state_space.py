import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def selective_scan(
    u: Float[Array, "seq_length d_inner"],
    delta: Float[Array, "seq_length d_inner"],
    A: Float[Array, "d_inner d_state"],
    B: Float[Array, "seq_length d_inner d_state"],
    C: Float[Array, "seq_length d_inner d_state"],
    D: Float[Array, " d_inner"],
    chunk_size: int = 128,
) -> Float[Array, "seq_length d_inner"]:
    deltaA = jnp.exp(jnp.einsum("l d, d n -> l d n", delta, A))
    deltaB_u = jnp.einsum("l d, l d n, l d -> l d n", delta, B, u)

    seq_len, d_inner = u.shape
    d_state = A.shape[1]

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    padded_len = num_chunks * chunk_size

    pad_len = padded_len - seq_len
    deltaA_padded = jnp.pad(deltaA, ((0, pad_len), (0, 0), (0, 0)))
    deltaB_u_padded = jnp.pad(deltaB_u, ((0, pad_len), (0, 0), (0, 0)))
    C_padded = jnp.pad(C, ((0, pad_len), (0, 0), (0, 0)))

    deltaA_chunked = deltaA_padded.reshape(num_chunks, chunk_size, d_inner, d_state)
    deltaB_u_chunked = deltaB_u_padded.reshape(num_chunks, chunk_size, d_inner, d_state)
    C_chunked = C_padded.reshape(num_chunks, chunk_size, d_inner, d_state)

    def intra_chunk_step(h_prev, scan_inputs):
        deltaA_i, deltaB_u_i, C_i = scan_inputs
        h_i = deltaA_i * h_prev + deltaB_u_i
        y_i = jnp.einsum("d n, d n -> d", h_i, C_i)
        return h_i, y_i

    h0 = jnp.zeros((d_inner, d_state))

    _, y_chunks = jax.vmap(jax.lax.scan, in_axes=(None, None, 0))(
        intra_chunk_step, h0, (deltaA_chunked, deltaB_u_chunked, C_chunked)
    )

    def inter_chunk_step(carry_prev, scan_inputs):
        A_prev, h_prev = carry_prev
        deltaA_i, deltaB_u_i = scan_inputs

        A_new = deltaA_i * A_prev
        h_new = deltaA_i * h_prev + deltaB_u_i

        return (A_new, h_new), (A_new, h_new)

    A_carry_initial = jnp.ones((d_inner, d_state))
    h_carry_initial = jnp.zeros((d_inner, d_state))
    initial_carry = (A_carry_initial, h_carry_initial)

    scan_inputs = (deltaA_chunked[:, -1], deltaB_u_chunked[:, -1])

    _, (A_carry, h_carry) = jax.lax.scan(inter_chunk_step, initial_carry, scan_inputs)

    A_carry = jnp.roll(A_carry, 1, axis=0)
    h_carry = jnp.roll(h_carry, 1, axis=0)
    A_carry = A_carry.at[0].set(jnp.ones_like(A_carry[0]))
    h_carry = h_carry.at[0].set(jnp.zeros_like(h_carry[0]))

    h_carry_broadcast = jnp.expand_dims(h_carry, axis=1)
    h_correction = deltaA_chunked * h_carry_broadcast
    y_carry = jnp.einsum("csdn, csdn -> csd", C_chunked, h_correction)

    y_final = y_chunks + y_carry

    y_final = y_final.reshape(padded_len, d_inner)

    y_unpadded = y_final[:seq_len]

    output = y_unpadded.real + u * D

    return output
