import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxonlayers.functions.masking import make_causal_mask
from jaxonlayers.layers import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TestTransformerEncoderLayer:
    @pytest.mark.parametrize(
        "is_causal, use_explicit_mask",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_masking(self, is_causal, use_explicit_mask):
        d_model = 64
        n_heads = 4
        seq_len = 10

        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        x = jax.random.normal(jax.random.key(1), (seq_len, d_model))
        mask = make_causal_mask(seq_len) if use_explicit_mask else None
        output = layer(x, attn_mask=mask, is_causal=is_causal, inference=True)

        assert output.shape == (seq_len, d_model)

    def test_jit_no_retrace(self):
        d_model = 64
        n_heads = 4
        seq_len = 10

        layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def forward(model, x, mask, is_causal, key, inference):
            return model(x, mask, is_causal, key, inference=inference)

        x = jax.random.normal(jax.random.key(1), (seq_len, d_model))

        output1 = forward(layer, x, None, False, None, True)
        output2 = forward(layer, x, None, False, None, True)
        output3 = forward(layer, x, None, False, None, True)

        assert output1.shape == (seq_len, d_model)
        assert output2.shape == (seq_len, d_model)
        assert output3.shape == (seq_len, d_model)


class TestTransformerDecoderLayer:
    @pytest.mark.parametrize(
        "tgt_is_causal, memory_is_causal, use_tgt_mask, use_memory_mask",
        [
            (False, False, False, False),
            (True, False, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, False, True, True),
        ],
    )
    def test_masking(
        self, tgt_is_causal, memory_is_causal, use_tgt_mask, use_memory_mask
    ):
        d_model = 64
        n_heads = 4
        tgt_len = 10
        src_len = 12

        layer = TransformerDecoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))
        tgt_mask = make_causal_mask(tgt_len) if use_tgt_mask else None
        memory_mask = (
            jnp.ones((tgt_len, src_len), dtype=jnp.bool_) if use_memory_mask else None
        )

        output = layer(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    def test_with_process_heads(self):
        d_model = 64
        n_heads = 4
        tgt_len = 10
        src_len = 12

        layer = TransformerDecoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        def identity_process_heads(q, k, v):
            return q, k, v

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))
        output = layer(
            tgt,
            memory,
            process_heads_sa=identity_process_heads,
            process_heads_mha=identity_process_heads,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    def test_jit_no_retrace(self):
        d_model = 64
        n_heads = 4
        tgt_len = 10
        src_len = 12

        layer = TransformerDecoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def forward(model, tgt, memory, inference):
            return model(tgt, memory, inference=inference)

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))

        output1 = forward(layer, tgt, memory, True)
        output2 = forward(layer, tgt, memory, True)
        output3 = forward(layer, tgt, memory, True)

        assert output1.shape == (tgt_len, d_model)
        assert output2.shape == (tgt_len, d_model)
        assert output3.shape == (tgt_len, d_model)


class TestTransformerEncoder:
    @pytest.mark.parametrize(
        "is_causal, use_explicit_mask",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_masking(self, is_causal, use_explicit_mask):
        d_model = 64
        n_heads = 4
        num_layers = 3
        seq_len = 10

        encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        x = jax.random.normal(jax.random.key(1), (seq_len, d_model))
        mask = make_causal_mask(seq_len) if use_explicit_mask else None
        output = encoder(x, mask=mask, is_causal=is_causal, inference=True)

        assert output.shape == (seq_len, d_model)

    def test_with_process_heads(self):
        d_model = 64
        n_heads = 4
        num_layers = 3
        seq_len = 10

        encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        def identity_process_heads(q, k, v):
            return q, k, v

        x = jax.random.normal(jax.random.key(1), (seq_len, d_model))
        output = encoder(x, process_heads=identity_process_heads, inference=True)

        assert output.shape == (seq_len, d_model)

    def test_jit_no_retrace(self):
        d_model = 64
        n_heads = 4
        num_layers = 3
        seq_len = 10

        encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def forward(model, x, inference):
            return model(x, inference=inference)

        x = jax.random.normal(jax.random.key(1), (seq_len, d_model))

        output1 = forward(encoder, x, True)
        output2 = forward(encoder, x, True)
        output3 = forward(encoder, x, True)

        assert output1.shape == (seq_len, d_model)
        assert output2.shape == (seq_len, d_model)
        assert output3.shape == (seq_len, d_model)


class TestTransformerDecoder:
    @pytest.mark.parametrize(
        "tgt_is_causal, memory_is_causal, use_tgt_mask, use_memory_mask",
        [
            (False, False, False, False),
            (True, False, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, False, True, True),
        ],
    )
    def test_masking(
        self, tgt_is_causal, memory_is_causal, use_tgt_mask, use_memory_mask
    ):
        d_model = 64
        n_heads = 4
        num_layers = 3
        tgt_len = 10
        src_len = 12

        decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))
        tgt_mask = make_causal_mask(tgt_len) if use_tgt_mask else None
        memory_mask = (
            jnp.ones((tgt_len, src_len), dtype=jnp.bool_) if use_memory_mask else None
        )

        output = decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    def test_with_process_heads(self):
        d_model = 64
        n_heads = 4
        num_layers = 3
        tgt_len = 10
        src_len = 12

        decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        def identity_process_heads(q, k, v):
            return q, k, v

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))
        output = decoder(
            tgt,
            memory,
            process_heads_sa=identity_process_heads,
            process_heads_mha=identity_process_heads,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    def test_jit_no_retrace(self):
        d_model = 64
        n_heads = 4
        num_layers = 3
        tgt_len = 10
        src_len = 12

        decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def forward(model, tgt, memory, inference):
            return model(tgt, memory, inference=inference)

        tgt = jax.random.normal(jax.random.key(1), (tgt_len, d_model))
        memory = jax.random.normal(jax.random.key(2), (src_len, d_model))

        output1 = forward(decoder, tgt, memory, True)
        output2 = forward(decoder, tgt, memory, True)
        output3 = forward(decoder, tgt, memory, True)

        assert output1.shape == (tgt_len, d_model)
        assert output2.shape == (tgt_len, d_model)
        assert output3.shape == (tgt_len, d_model)


class TestTransformer:
    @pytest.mark.parametrize(
        "src_is_causal, tgt_is_causal, memory_is_causal, use_src_mask, use_tgt_mask, use_memory_mask",
        [
            (False, False, False, False, False, False),
            (True, True, False, False, False, False),
            (False, False, False, True, True, True),
            (True, True, False, True, True, True),
        ],
    )
    def test_masking(
        self,
        src_is_causal,
        tgt_is_causal,
        memory_is_causal,
        use_src_mask,
        use_tgt_mask,
        use_memory_mask,
    ):
        d_model = 64
        n_heads = 4
        src_len = 12
        tgt_len = 10

        transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        src = jax.random.normal(jax.random.key(1), (src_len, d_model))
        tgt = jax.random.normal(jax.random.key(2), (tgt_len, d_model))
        src_mask = make_causal_mask(src_len) if use_src_mask else None
        tgt_mask = make_causal_mask(tgt_len) if use_tgt_mask else None
        memory_mask = (
            jnp.ones((tgt_len, src_len), dtype=jnp.bool_) if use_memory_mask else None
        )

        output = transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_is_causal=src_is_causal,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    def test_with_process_heads(self):
        d_model = 64
        n_heads = 4
        src_len = 12
        tgt_len = 10

        transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        def identity_process_heads(q, k, v):
            return q, k, v

        src = jax.random.normal(jax.random.key(1), (src_len, d_model))
        tgt = jax.random.normal(jax.random.key(2), (tgt_len, d_model))
        output = transformer(
            src,
            tgt,
            process_heads_enc=identity_process_heads,
            process_heads_dec_sa=identity_process_heads,
            process_heads_dec_mha=identity_process_heads,
            inference=True,
        )

        assert output.shape == (tgt_len, d_model)

    @pytest.mark.parametrize(
        "activation",
        [
            jax.nn.relu,
            jax.nn.gelu,
            jax.nn.silu,
        ],
    )
    def test_activations(self, activation):
        d_model = 64
        n_heads = 4
        src_len = 12
        tgt_len = 10

        transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            activation=activation,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        src = jax.random.normal(jax.random.key(1), (src_len, d_model))
        tgt = jax.random.normal(jax.random.key(2), (tgt_len, d_model))
        output = transformer(src, tgt, inference=True)

        assert output.shape == (tgt_len, d_model)

    def test_jit_no_retrace(self):
        d_model = 64
        n_heads = 4
        src_len = 12
        tgt_len = 10

        transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            key=jax.random.key(0),
            dtype=jnp.float32,
        )

        @eqx.filter_jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def forward(model, src, tgt, inference):
            return model(src, tgt, inference=inference)

        src = jax.random.normal(jax.random.key(1), (src_len, d_model))
        tgt = jax.random.normal(jax.random.key(2), (tgt_len, d_model))

        output1 = forward(transformer, src, tgt, True)
        output2 = forward(transformer, src, tgt, True)
        output3 = forward(transformer, src, tgt, True)

        assert output1.shape == (tgt_len, d_model)
        assert output2.shape == (tgt_len, d_model)
        assert output3.shape == (tgt_len, d_model)
