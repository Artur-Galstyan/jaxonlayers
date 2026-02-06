import equinox as eqx
from beartype.typing import Any
from jaxtyping import Array, PRNGKeyArray

from jaxonlayers.functions import default_floating_dtype


class EmbeddingWithPadding(eqx.Module):
    embed: eqx.nn.Embedding
    padding_idx: int = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.embed = eqx.nn.Embedding(
            num_embeddings, embedding_dim, key=key, dtype=dtype
        )
        self.padding_idx = padding_idx

    def __call__(self, x: Array):
        out = self.embed(x)
        mask = (x != self.padding_idx).astype(out.dtype)
        return out * mask[..., None]


class EmbeddingBag(eqx.Module):
    embed: EmbeddingWithPadding

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.embed = EmbeddingWithPadding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, key=key, dtype=dtype
        )

    def __call__(self, x):
        looked_up = eqx.filter_vmap(self.embed)(x)
        return looked_up.sum(axis=0)
