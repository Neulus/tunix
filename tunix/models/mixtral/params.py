"""Utils for loading and converting Mixtral PT weights."""

import re

import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
import tqdm
from etils import epath
from flax import nnx

from tunix.models.mixtral import model as model_lib


def _stack_experts(params: dict[str, jax.Array]):
    """Stack experts in the loaded pytorch params."""
    key_fn = lambda x: int(re.match(r".*?\.experts\.([0-9]+)\..*", x).group(1))
    new_params = dict(params).copy()
    for kw in ["w1", "w2", "w3"]:
        pattern = r"(.*?)block_sparse_moe\.experts\.([0-9]+)\.{}\.weight".format(kw)
        keys = [k for k in params.keys() if re.match(pattern, k)]
        prefix_groups = set([re.match(pattern, k).group(1) for k in keys])
        for prefix in prefix_groups:
            keys_to_merge = sorted(
                [k for k in keys if k.startswith(prefix)], key=key_fn
            )
            for k in keys_to_merge:
                del new_params[k]
            with jax.default_device(jax.devices("cpu")[0]):
                new_key = f"{prefix}block_sparse_moe.experts.{kw}.weight"
                stacked_tensor = jnp.stack([params[k] for k in keys_to_merge], axis=0)
                new_params[new_key] = stacked_tensor
    return new_params


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        # Embeddings and Final Layer
        r"model\.embed_tokens\.weight": ("embed_tokens.input_embedding", None),
        r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
        # Attention projection weights
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.self_attn.q_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.self_attn.k_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.self_attn.v_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.self_attn.o_proj.w",
            ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
        ),
        # MoE Router (Gating network)
        r"model\.layers\.([0-9]+)\.block_sparse_moe\.gate\.weight": (
            r"layers.\1.block_sparse_moe.gate.kernel",
            ((1, 0), None),
        ),
        # Stacked MoE expert weights (after processing by _stack_mixtral_experts)
        r"model\.layers\.([0-9]+)\.block_sparse_moe\.experts\.w1\.weight": (
            r"layers.\1.block_sparse_moe.w1",
            ((0, 2, 1), None),
        ),
        r"model\.layers\.([0-9]+)\.block_sparse_moe\.experts\.w3\.weight": (
            r"layers.\1.block_sparse_moe.w3",
            ((0, 2, 1), None),
        ),
        r"model\.layers\.([0-9]+)\.block_sparse_moe\.experts\.w2\.weight": (
            r"layers.\1.block_sparse_moe.w2",
            ((0, 2, 1), None),
        ),
        # RMS Norms
        r"model\.norm\.weight": ("norm.w", None),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.w",
            None,
        ),
    }


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 1:
        return subs[0]
    elif len(subs) == 0:
        raise ValueError(f"No matching key pattern found for: {source_key}")
    else:
        raise ValueError(f"Multiple key patterns matched for: {source_key} -> {subs}")


def _assign_weights(keys, tensor, state_dict, torch_key, transform):
    """Convert weights and assign to nnx state_dict."""
    key = keys[0]
    if len(keys) == 1:
        try:
            if transform:
                permute, reshape = transform
                tensor = tensor.transpose(permute) if permute else tensor
                tensor = tensor.reshape(reshape) if reshape else tensor
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform tensor {torch_key} with shape {tensor.shape}: {e}"
            ) from e

        if key not in state_dict:
            raise ValueError(
                f"{key} does not exist in {torch_key} -> {'.'.join(map(str, keys))}; Only {state_dict}."
            )

        if tensor.shape != state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for {torch_key} -> {'.'.join(map(str, keys))}. "
                f"Got {tensor.shape}, expected {state_dict[key].shape}."
            )
        state_dict[key] = tensor
        return state_dict
    else:
        if key not in state_dict:
            raise ValueError(f"Key '{key}' not found in JAX model state.")
        _assign_weights(keys[1:], tensor, state_dict[key], torch_key, transform)
        return state_dict


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Mixtral:
    """Load tensors from the safetensors file and create a Mixtral model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

    if not files:
        raise ValueError(f"No .safetensors files found in {file_dir}")

    mixtral = nnx.eval_shape(lambda: model_lib.Mixtral(config, rngs=nnx.Rngs(params=0)))

    graph_def, abs_state = nnx.split(mixtral)
    state_dict = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping(config)

    with jax.default_device(jax.devices("cpu")[0]):
        tensor_dict = {}

        for f in tqdm.tqdm(files):
            tensor_dict |= safetensors.load_file(f)

        if config.num_experts is not None:
            tensor_dict = _stack_experts(tensor_dict)

        for k, v in tqdm.tqdm(tensor_dict.items()):
            try:
                jax_key_str, transform = _torch_key_to_jax_key(mapping, k)
                jax_keys = [_stoi(s) for s in jax_key_str.split(".")]
                _assign_weights(jax_keys, v, state_dict, k, transform)
            except ValueError as e:
                print(f"Skipping key '{k}': {e}")
                continue

    if mesh:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        state_dict = jax.device_put(state_dict, sharding)
    else:
        state_dict = jax.device_put(state_dict, jax.devices()[0])

    return nnx.merge(graph_def, state_dict)
