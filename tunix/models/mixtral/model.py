import dataclasses
from typing import Tuple

import flax
import flax.typing
import jax
import jax.numpy as jnp
import jax.sharding as shd
import jaxtyping
from flax import nnx
from jax.interpreters import pxla

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for Qwen3 model."""

    emb_vd: Tuple[str | None, ...]
    emb_dv: Tuple[str | None, ...]
    q_weight_ndh: Tuple[str | None, ...]
    kv_weight_ndh: Tuple[str | None, ...]
    o_weight_nhd: Tuple[str | None, ...]
    ffw_weight_df: Tuple[str | None, ...]
    ffw_weight_fd: Tuple[str | None, ...]
    rms_norm_weight: Tuple[str | None, ...]
    act_btd: Tuple[str | None, ...]
    act_btf: Tuple[str | None, ...]
    act_btnh: Tuple[str | None, ...]
    exp_weight_cdf: Tuple[str | None, ...]
    exp_weight_cfd: Tuple[str | None, ...]

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None

        return ShardingConfig(
            emb_vd=("tp", fsdp),
            emb_dv=(fsdp, "tp"),
            q_weight_ndh=("tp", fsdp, None),
            kv_weight_ndh=("tp", fsdp, None),
            o_weight_nhd=("tp", None, fsdp),
            ffw_weight_df=(fsdp, "tp"),
            ffw_weight_fd=("tp", fsdp),
            rms_norm_weight=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
            exp_weight_cdf=("fsdp", None, "tp"),
            exp_weight_cfd=("fsdp", "tp", None),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Mixtral model."""

    num_layers: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()


class Embedder(nnx.Module):
    """Embedder module."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.input_embedding = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
            sharding=shd_config.emb_vd,
        )
        self.shd_config = shd_config

    @jax.named_scope("embedder_encode")
    def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        x = self.input_embedding[(x,)]
        x = shard(x, self.shd_config.act_btd)  # type: ignore
        return x  # type: ignore

    @jax.named_scope("embedder_decode")
    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.dot(x, self.input_embedding.value.T)


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        dim: int,
        *,
        norm_eps: float = 1e-06,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.w = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), dim),  # type: ignore
            sharding=shd_config.rms_norm_weight,
        )
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        rms = jnp.sqrt(
            jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
            + self.norm_eps
        )
        return jnp.astype(self.w * x / rms, dtype)


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == "cpu":
        return x
    return jax.lax.with_sharding_constraint(
        x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
    )


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
    """Applies RoPE."""
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class Einsum(nnx.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(
        self,
        einsum_str: str,
        shape: flax.typing.Shape,
        *,
        rngs: nnx.Rngs,
        sharding: Tuple[str | None, ...],
    ):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = nnx.Param(
            nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
        )

    @jax.named_scope("einsum")
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


class Attention(nnx.Module):
    """Attention module."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.shd_config = shd_config
        self.wq = Einsum(
            einsum_str="BTD,DNH->BTNH",
            shape=(
                config.embed_dim,
                config.num_heads,
                config.head_dim,
            ),  # convert to (e, n * h) -. (e, n, h) in params.py
            rngs=rngs,
            sharding=shd_config.q_weight_ndh,
        )
        self.wk = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )
        self.wv = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )
        self.wo = Einsum(
            einsum_str="BTNH,NHD->BTD",
            shape=(config.num_heads, config.head_dim, config.embed_dim),
            rngs=rngs,
            sharding=shd_config.o_weight_nhd,
        )
        self.n_rep = config.num_heads // config.num_kv_heads
        self.scale = self.head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(
        self,
        x: jaxtyping.Array,
        segment_pos: jaxtyping.Array,
        cache: LayerCache | None,
        attn_mask: jaxtyping.Array | None,
    ) -> tuple[LayerCache | None, jaxtyping.Array]:
        seq_len = x.shape[1]

        query_proj = self.wq(x)
        key_proj = self.wk(x)
        value_proj = self.wv(x)

        query_proj = shard(query_proj, self.shd_config.act_btnh)  # type: ignore
        key_proj = shard(key_proj, self.shd_config.act_btnh)  # type: ignore
        value_proj = shard(value_proj, self.shd_config.act_btnh)  # type: ignore

        query_proj = apply_rope(
            query_proj,
            segment_pos,
            head_dim=self.head_dim,
        )
        key_proj = apply_rope(
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
        )

        if cache is not None:
            end_index = cache["end_index"][0]
            slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
            value_proj = jax.lax.dynamic_update_slice(
                cache["v"],
                value_proj,
                slice_indices,
            )
            key_proj = jax.lax.dynamic_update_slice(cache["k"], key_proj, slice_indices)

        attn = jnp.einsum("BTHD,BSHD->BHTS", query_proj, key_proj) * self.scale

        if attn_mask is not None:
            attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(key_proj.dtype)  # type: ignore

        qkv = jnp.einsum("BHTS,BSHD->BTHD", attn, value_proj)

        outputs = self.wo(qkv)
        outputs = shard(outputs, self.shd_config.act_btd)  # type: ignore

        if cache is not None:
            new_cache = {
                "v": value_proj,
                "k": key_proj,
                "end_index": cache["end_index"] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, outputs

    @property
    def head_dim(self):
        return self.wo.shape[1]

    @property
    def num_heads(self):
        return self.wq.shape[0]

    @property
    def num_kv_heads(self):
        return self.wk.shape[1]


class MLP(nnx.Module):
    """MLP module."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.shd_config = shd_config
        kernel_init_fn = nnx.initializers.zeros_init()
        self.w1 = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init_fn, shd_config.ffw_weight_df),
        )
        self.w2 = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.embed_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init_fn, shd_config.ffw_weight_fd),
        )
        self.w3 = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init_fn, shd_config.ffw_weight_df),
        )

    @jax.named_scope("feed_forward")
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        activations = nnx.silu(self.w1(x)) * self.w3(x)  # type: ignore
        activations = shard(activations, self.shd_config.act_btf)  # type: ignore
        outputs = self.w2(activations)
        return outputs


class MOEFeedForward(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.shd_config = shd_config

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = [
            MLP(config, rngs=rngs, shd_config=shd_config)
            for _ in range(self.num_experts)
        ]
        self.gate = nnx.Linear(
            config.embed_dim,
            self.num_experts,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x):
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])  # [B*T, D]

        # Compute gating scores and top-k selection
        gates = self.gate(x)
        scores, inds = jax.lax.top_k(gates, k=ne)
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)

        # Efficient conditional computation using segment_sum
        # First, create flattened indices and inputs
        batch_indices = jnp.arange(x.shape[0])[:, None]  # [B*T, 1]
        batch_indices = jnp.tile(batch_indices, (1, ne)).reshape(-1)  # [B*T*ne]

        expert_indices = inds.reshape(-1)  # [B*T*ne]
        scores_flat = scores.reshape(-1)  # [B*T*ne]

        # Replicate inputs for each selected expert
        x_replicated = jnp.repeat(x, ne, axis=0)  # [B*T*ne, D]

        # Apply experts conditionally
        def apply_expert(carry, idx):
            expert_idx, token_idx, score, token_x = idx
            expert_out = jax.lax.switch(
                expert_idx,
                [lambda x: self.experts[j](x) for j in range(self.num_experts)],
                token_x,
            )
            return carry, expert_out * score

        _, expert_outputs = jax.lax.scan(
            apply_expert,
            None,
            (expert_indices, batch_indices, scores_flat, x_replicated),
        )

        # Sum contributions for each token
        y = jax.ops.segment_sum(expert_outputs, batch_indices, num_segments=x.shape[0])

        return y.reshape(orig_shape)


class DecoderLayer(nnx.Module):
    """DecoderLayer."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.attention_norm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.attention = Attention(
            config=config,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.feed_forward = MOEFeedForward(
            config=config,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.ffn_norm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            shd_config=shd_config,
        )

    def __call__(
        self,
        x: jaxtyping.Array,
        segment_pos: jaxtyping.Array,
        cache: LayerCache | None,
        attn_mask: jaxtyping.Array,
    ) -> tuple[LayerCache | None, jaxtyping.Array]:
        inputs_normalized = self.attention_norm(x)
        cache, attn_output = self.attention(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )
        attn_output += x
        residual = attn_output
        attn_output = self.ffn_norm(attn_output)
        outputs = residual + self.feed_forward(attn_output)
        return cache, outputs


class Mixtral(nnx.Module):
    """Mixtral model."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.config = config
        self.embedder = Embedder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.layers = [
            DecoderLayer(config=config, rngs=rngs, shd_config=shd_config)
            for _ in range(config.num_layers)
        ]
        self.norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
        )
        self.output = Einsum(
            einsum_str="BTD,DV->BTV",
            shape=(config.embed_dim, config.vocab_size),
            rngs=rngs,
            sharding=shd_config.emb_dv,
        )

    def get_model_input(self):
        """Returns a dummy model input for the transformer."""
        dummy_batch_size = 1
        dummy_seq_len = 128
        return {
            "input_tokens": jnp.ones(
                (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
            ),
            "positions": jnp.ones((dummy_batch_size, dummy_seq_len), dtype=jnp.int32),
            "cache": None,
            "attention_mask": jnp.ones(
                (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
            ),
        }

    def __call__(
        self,
        input_tokens: jaxtyping.Array,  # [B, L]
        positions: jaxtyping.Array,  # [B, L]
        cache: Cache | None,  # (sequence length L')
        attention_mask: jaxtyping.Array,  # [B, L, L']
    ) -> tuple[jaxtyping.Array, Cache | None]:
        """Mixtral model.

        Args:
          input_tokens: input sequence of tokens.
          positions: input absolute positions.
          cache: Attention KV cache or None.
          attention_mask: transformer input mask.

        Returns:
          predicted_logits, new_cache

          predicted_logits: output logits predicted by the model
          new_cache: updated cache if the input cache is not None, None elsewhere.
        """
        new_cache = None if cache is None else {}
        x = self.embedder.encode(input_tokens)

        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i}"
            layer_cache = cache[layer_name] if cache else None
            layer_cache, x = layer(
                x,
                positions,
                layer_cache,
                attention_mask,
            )
            if cache is not None and new_cache is not None:
                new_cache[layer_name] = (
                    layer_cache  # pytype: disable=container-type-mismatch
                )

        x = self.norm(x)
        logits = self.output(x)

        return logits, new_cache  # pytype: disable=bad-return-type

    @property
    def num_embed(self) -> int:
        return self.config.embed_dim
