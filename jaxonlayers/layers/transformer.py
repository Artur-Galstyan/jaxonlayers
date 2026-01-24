from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxonlayers.functions import default_floating_dtype
from jaxonlayers.functions.masking import make_causal_mask

_ProcessHeads = Callable[
    [
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ],
    tuple[
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ],
]


class TransformerEncoderLayer(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    norm_first: bool
    activation: Callable
    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: Callable = jax.nn.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        use_bias=True,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference
        mha_key, lin1_key, lin2_key = jax.random.split(key, 3)
        self.self_attn = eqx.nn.MultiheadAttention(
            n_heads,
            d_model,
            dropout_p=dropout_p,
            use_query_bias=use_bias,
            use_key_bias=use_bias,
            use_value_bias=use_bias,
            use_output_bias=use_bias,
            inference=inference,
            key=mha_key,
            dtype=dtype,
        )
        self.linear1 = eqx.nn.Linear(
            d_model, dim_feedforward, use_bias=use_bias, dtype=dtype, key=lin1_key
        )
        self.dropout = eqx.nn.Dropout(dropout_p, inference=inference)
        self.linear2 = eqx.nn.Linear(
            dim_feedforward, d_model, use_bias=use_bias, dtype=dtype, key=lin2_key
        )

        self.norm_first = norm_first
        self.norm1 = eqx.nn.LayerNorm(
            d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
        )
        self.norm2 = eqx.nn.LayerNorm(
            d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
        )

        self.dropout1 = eqx.nn.Dropout(dropout_p, inference=inference)
        self.dropout2 = eqx.nn.Dropout(dropout_p, inference=inference)
        self.activation = activation

    def _sa_block(
        self,
        x: Float[Array, "seq_len d_model"],
        attn_mask: Bool[Array, "seq_len seq_len"] | None,
        is_causal: bool,
        key: PRNGKeyArray | None,
        process_heads: _ProcessHeads | None = None,
        inference: bool = False,
    ) -> Float[Array, "seq_len d_model"]:
        seq_len, _ = x.shape
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        if is_causal:
            causal_mask = make_causal_mask(seq_len)
            if attn_mask is not None:
                mask = attn_mask & causal_mask
            else:
                mask = causal_mask
        else:
            mask = attn_mask

        x = self.self_attn(
            x,
            x,
            x,
            mask=mask,
            key=key1,
            process_heads=process_heads,
            inference=inference,
        )
        return self.dropout1(x, inference=inference, key=key2)

    def _ff_block(
        self,
        x: Float[Array, "seq_len d_model"],
        key: PRNGKeyArray | None,
        inference: bool = False,
    ):
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None
        x = eqx.filter_vmap(self.linear2)(
            self.dropout(
                self.activation(eqx.filter_vmap(self.linear1)(x)),
                inference=inference,
                key=key1,
            )
        )
        return self.dropout2(x, inference=inference, key=key2)

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        attn_mask: Bool[Array, "seq_len seq_len"] | None = None,
        is_causal: bool = False,
        key: PRNGKeyArray | None = None,
        process_heads: _ProcessHeads | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "seq_len d_model"]:
        if inference is None:
            inference = self.inference
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        if self.norm_first:
            x = x + self._sa_block(
                eqx.filter_vmap(self.norm1)(x),
                attn_mask,
                is_causal,
                key1,
                process_heads,
                inference=inference,
            )
            x = x + self._ff_block(
                eqx.filter_vmap(self.norm2)(x), key2, inference=inference
            )
        else:
            x = eqx.filter_vmap(self.norm1)(
                x
                + self._sa_block(
                    x, attn_mask, is_causal, key1, process_heads, inference=inference
                )
            )
            x = eqx.filter_vmap(self.norm2)(
                x + self._ff_block(x, key2, inference=inference)
            )

        return x


class TransformerDecoderLayer(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    multihead_attn: eqx.nn.MultiheadAttention

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout

    norm_first: bool
    activation: Callable

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: Callable = jax.nn.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        use_bias: bool = True,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        mha_key1, mha_key2, lin1_key, lin2_key = jax.random.split(key, 4)
        self.self_attn = eqx.nn.MultiheadAttention(
            n_heads,
            d_model,
            dropout_p=dropout_p,
            use_query_bias=use_bias,
            use_key_bias=use_bias,
            use_value_bias=use_bias,
            use_output_bias=use_bias,
            inference=inference,
            key=mha_key1,
            dtype=dtype,
        )
        self.multihead_attn = eqx.nn.MultiheadAttention(
            n_heads,
            d_model,
            dropout_p=dropout_p,
            use_query_bias=use_bias,
            use_key_bias=use_bias,
            use_value_bias=use_bias,
            use_output_bias=use_bias,
            inference=inference,
            key=mha_key2,
            dtype=dtype,
        )

        self.linear1 = eqx.nn.Linear(
            d_model,
            dim_feedforward,
            use_bias=use_bias,
            key=lin1_key,
            dtype=dtype,
        )
        self.dropout = eqx.nn.Dropout(dropout_p, inference=inference)
        self.linear2 = eqx.nn.Linear(
            dim_feedforward, d_model, use_bias=use_bias, key=lin2_key, dtype=dtype
        )

        self.norm_first = norm_first
        self.norm1 = eqx.nn.LayerNorm(
            d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
        )
        self.norm2 = eqx.nn.LayerNorm(
            d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
        )
        self.norm3 = eqx.nn.LayerNorm(
            d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
        )
        self.dropout1 = eqx.nn.Dropout(dropout_p, inference=inference)
        self.dropout2 = eqx.nn.Dropout(dropout_p, inference=inference)
        self.dropout3 = eqx.nn.Dropout(dropout_p, inference=inference)

        self.activation = activation

    # self-attention block
    def _sa_block(
        self,
        x: Float[Array, "seq_len d_model"],
        attn_mask: Bool[Array, "seq_len seq_len"] | None,
        is_causal: bool,
        key: PRNGKeyArray | None,
        process_heads: _ProcessHeads | None = None,
        inference: bool = False,
    ) -> Float[Array, "seq_len d_model"]:
        seq_len, _ = x.shape
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        if is_causal:
            causal_mask = make_causal_mask(seq_len)
            if attn_mask is not None:
                mask = attn_mask & causal_mask
            else:
                mask = causal_mask
        else:
            mask = attn_mask

        x = self.self_attn(
            x,
            x,
            x,
            mask=mask,
            key=key1,
            process_heads=process_heads,
            inference=inference,
        )
        return self.dropout1(x, inference=inference, key=key2)

    # multihead attention block
    def _mha_block(
        self,
        x: Float[Array, "tgt_len d_model"],
        mem: Float[Array, "src_len d_model"],
        attn_mask: Bool[Array, "tgt_len src_len"] | None,
        is_causal: bool,
        key: PRNGKeyArray | None,
        process_heads: _ProcessHeads | None = None,
        inference: bool = False,
    ) -> Float[Array, "tgt_len d_model"]:
        tgt_len, _ = x.shape
        src_len, _ = mem.shape
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        if is_causal:
            causal_mask = jnp.tril(jnp.ones((tgt_len, src_len), dtype=jnp.bool_))
            if attn_mask is not None:
                mask = attn_mask & causal_mask
            else:
                mask = causal_mask
        else:
            mask = attn_mask

        x = self.multihead_attn(
            x,
            mem,
            mem,
            mask=mask,
            key=key1,
            process_heads=process_heads,
            inference=inference,
        )
        return self.dropout2(x, inference=inference, key=key2)

    # feed forward block
    def _ff_block(
        self,
        x: Float[Array, "seq_len d_model"],
        key: PRNGKeyArray | None,
        inference: bool = False,
    ) -> Float[Array, "seq_len d_model"]:
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1, key2 = None, None

        x = eqx.filter_vmap(self.linear2)(
            self.dropout(
                self.activation(eqx.filter_vmap(self.linear1)(x)),
                inference=inference,
                key=key1,
            )
        )
        return self.dropout3(x, inference=inference, key=key2)

    def __call__(
        self,
        tgt: Float[Array, "tgt_len d_model"],
        memory: Float[Array, "src_len d_model"],
        tgt_mask: Bool[Array, "tgt_len tgt_len"] | None = None,
        memory_mask: Bool[Array, "tgt_len src_len"] | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        key: PRNGKeyArray | None = None,
        process_heads_sa: _ProcessHeads | None = None,
        process_heads_mha: _ProcessHeads | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "tgt_len d_model"]:
        if inference is None:
            inference = self.inference
        if key is not None:
            key1, key2, key3 = jax.random.split(key, 3)
        else:
            key1, key2, key3 = None, None, None

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                eqx.filter_vmap(self.norm1)(x),
                tgt_mask,
                tgt_is_causal,
                key1,
                process_heads_sa,
                inference=inference,
            )
            x = x + self._mha_block(
                eqx.filter_vmap(self.norm2)(x),
                memory,
                memory_mask,
                memory_is_causal,
                key2,
                process_heads_mha,
                inference=inference,
            )
            x = x + self._ff_block(
                eqx.filter_vmap(self.norm3)(x), key3, inference=inference
            )
        else:
            x = eqx.filter_vmap(self.norm1)(
                x
                + self._sa_block(
                    x,
                    tgt_mask,
                    tgt_is_causal,
                    key1,
                    process_heads_sa,
                    inference=inference,
                )
            )
            x = eqx.filter_vmap(self.norm2)(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_is_causal,
                    key2,
                    process_heads_mha,
                    inference=inference,
                )
            )
            x = eqx.filter_vmap(self.norm3)(
                x + self._ff_block(x, key3, inference=inference)
            )

        return x


class TransformerEncoder(eqx.Module):
    layers: list
    norm: eqx.nn.LayerNorm | None
    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: Callable = jax.nn.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        use_bias: bool = True,
        use_final_norm: bool = True,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        keys = jax.random.split(key, num_layers)
        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout_p=dropout_p,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                use_bias=use_bias,
                key=k,
                dtype=dtype,
                inference=inference,
            )
            for k in keys
        ]

        if use_final_norm:
            self.norm = eqx.nn.LayerNorm(
                d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
            )
        else:
            self.norm = None

    def __call__(
        self,
        src: Float[Array, "seq_len d_model"],
        mask: Bool[Array, "seq_len seq_len"] | None = None,
        is_causal: bool = False,
        key: PRNGKeyArray | None = None,
        process_heads: _ProcessHeads | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "seq_len d_model"]:
        if inference is None:
            inference = self.inference

        if key is not None:
            keys = jax.random.split(key, len(self.layers))
        else:
            keys = [None] * len(self.layers)

        x = src
        for layer, k in zip(self.layers, keys):
            x = layer(x, mask, is_causal, k, process_heads, inference=inference)

        if self.norm is not None:
            x = eqx.filter_vmap(self.norm)(x)

        return x


class TransformerDecoder(eqx.Module):
    layers: list
    norm: eqx.nn.LayerNorm | None
    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: Callable = jax.nn.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        use_bias: bool = True,
        use_final_norm: bool = True,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        keys = jax.random.split(key, num_layers)
        self.layers = [
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout_p=dropout_p,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                use_bias=use_bias,
                key=k,
                dtype=dtype,
                inference=inference,
            )
            for k in keys
        ]

        if use_final_norm:
            self.norm = eqx.nn.LayerNorm(
                d_model, eps=layer_norm_eps, use_bias=use_bias, dtype=dtype
            )
        else:
            self.norm = None

    def __call__(
        self,
        tgt: Float[Array, "tgt_len d_model"],
        memory: Float[Array, "src_len d_model"],
        tgt_mask: Bool[Array, "tgt_len tgt_len"] | None = None,
        memory_mask: Bool[Array, "tgt_len src_len"] | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        key: PRNGKeyArray | None = None,
        process_heads_sa: _ProcessHeads | None = None,
        process_heads_mha: _ProcessHeads | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "tgt_len d_model"]:
        if inference is None:
            inference = self.inference
        if key is not None:
            keys = jax.random.split(key, len(self.layers))
        else:
            keys = [None] * len(self.layers)

        x = tgt
        for layer, k in zip(self.layers, keys):
            x = layer(
                x,
                memory,
                tgt_mask,
                memory_mask,
                tgt_is_causal,
                memory_is_causal,
                k,
                process_heads_sa,
                process_heads_mha,
                inference=inference,
            )

        if self.norm is not None:
            x = eqx.filter_vmap(self.norm)(x)

        return x


class Transformer(eqx.Module):
    encoder: TransformerEncoder
    decoder: TransformerDecoder
    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: Callable = jax.nn.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        use_bias: bool = True,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        enc_key, dec_key = jax.random.split(key)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_p=dropout_p,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            use_bias=use_bias,
            use_final_norm=True,
            key=enc_key,
            dtype=dtype,
            inference=inference,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_p=dropout_p,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            use_bias=use_bias,
            use_final_norm=True,
            key=dec_key,
            dtype=dtype,
            inference=inference,
        )

    def __call__(
        self,
        src: Float[Array, "src_len d_model"],
        tgt: Float[Array, "tgt_len d_model"],
        src_mask: Bool[Array, "src_len src_len"] | None = None,
        tgt_mask: Bool[Array, "tgt_len tgt_len"] | None = None,
        memory_mask: Bool[Array, "tgt_len src_len"] | None = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        key: PRNGKeyArray | None = None,
        process_heads_enc: _ProcessHeads | None = None,
        process_heads_dec_sa: _ProcessHeads | None = None,
        process_heads_dec_mha: _ProcessHeads | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "tgt_len d_model"]:
        if inference is None:
            inference = self.inference
        if key is not None:
            enc_key, dec_key = jax.random.split(key)
        else:
            enc_key, dec_key = None, None

        memory = self.encoder(
            src,
            src_mask,
            src_is_causal,
            enc_key,
            process_heads_enc,
            inference=inference,
        )

        output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_is_causal,
            memory_is_causal,
            dec_key,
            process_heads_dec_sa,
            process_heads_dec_mha,
            inference=inference,
        )

        return output
