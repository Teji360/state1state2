"""
KV cache compression via frozen-basis SVD.

Strategy: register a custom attention implementation in transformers'
ALL_ATTENTION_FUNCTIONS global registry, then set model.config._attn_implementation.
Works for any model that dispatches through that registry (GPT-2, LLaMA, Mistral, etc.).
Handles Grouped Query Attention (GQA) — K/V heads are compressed once per KV group,
then each Q head in that group attends to the same compressed state.
"""

import math
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Optional, List, Tuple

from kv_state import cold_start, frozen_cold_start, sinked_cold_start
from attention import compressed_attention_with_weights, frozen_compressed_attention, sinked_attention

# PyramidKV-style per-layer rank budget for GPT-2 (12 layers).
# Lower layers see diffuse attention → larger rank; upper layers focus sharply → smaller rank.
PYRAMID_RANKS = [32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4]

# 32-layer models (LLaMA-2-7B, Mistral-7B).
PYRAMID_RANKS_32L = [32] * 8 + [16] * 8 + [8] * 8 + [4] * 8

# 22-layer models (TinyLlama-1.1B).
PYRAMID_RANKS_22L = [32] * 6 + [16] * 6 + [8] * 6 + [4] * 4


def torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.array(t.detach().float().cpu().numpy())


def jax_to_torch(x: jnp.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.array(x), device=device, dtype=torch.float32)


def _make_compressed_attn_fn(phi: jnp.ndarray, rank: int, T_max: int):
    """
    Returns a drop-in for transformers' eager_attention_forward.
    Signature matches ALL_ATTENTION_FUNCTIONS interface:
      fn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs)
    Returns (attn_output, attn_weights) where attn_output is
      (batch, seq_len, heads, head_dim) — same as eager_attention_forward.
    """
    def _compressed_attn(
        module,
        query: torch.Tensor,   # (batch, heads, seq_len, head_dim)
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch, n_heads, _, head_dim = key.shape
        device = query.device

        out_heads = []
        for b in range(batch):
            out_batch = []
            for h in range(n_heads):
                k_h = torch_to_jax(key[b, h])    # (seq_len, head_dim)
                v_h = torch_to_jax(value[b, h])  # (seq_len, head_dim)
                q_h = torch_to_jax(query[b, h])  # (seq_len, head_dim)

                state = cold_start(k_h, v_h, phi, rank, T_max, head_dim)
                # compressed_attention_with_weights runs the Moment-KV EMA update
                # on state.importance as a side-product of the attention computation.
                out_h, _ = compressed_attention_with_weights(q_h, state, T_max, causal=True)
                out_batch.append(jax_to_torch(out_h, device))  # (seq_len, head_dim)

            # (heads, seq_len, head_dim)
            out_heads.append(torch.stack(out_batch, dim=0))

        # (batch, heads, seq_len, head_dim) → transpose to (batch, seq_len, heads, head_dim)
        attn_output = torch.stack(out_heads, dim=0).transpose(1, 2)
        return attn_output, None

    return _compressed_attn


def _make_frozen_pyramid_fn(pyramid_ranks):
    """
    Returns a drop-in for transformers' eager_attention_forward that uses
    frozen-basis SVD compression with a per-layer rank budget.

    The rank for each call is read from module.layer_idx, so a single
    registered function handles all layers correctly.
    """
    def _frozen_pyramid_attn(
        module,
        query: torch.Tensor,   # (batch, heads, seq_len, head_dim)
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        rank = pyramid_ranks[min(layer_idx, len(pyramid_ranks) - 1)]
        batch, n_heads, _, _ = key.shape
        device = query.device

        out_heads = []
        for b in range(batch):
            out_batch = []
            for h in range(n_heads):
                k_h = torch_to_jax(key[b, h])
                v_h = torch_to_jax(value[b, h])
                q_h = torch_to_jax(query[b, h])

                state = frozen_cold_start(k_h, v_h, rank)
                out_h = frozen_compressed_attention(q_h, state, causal=True)
                out_batch.append(jax_to_torch(out_h, device))

            out_heads.append(torch.stack(out_batch, dim=0))

        attn_output = torch.stack(out_heads, dim=0).transpose(1, 2)
        return attn_output, None

    return _frozen_pyramid_attn


def _make_hybrid_pyramid_fn(warmup_len, warmup_rank, pyramid_ranks):
    """
    Hybrid: Brand's SVD (full-sequence, optimal basis) for seq_len <= warmup_len;
    frozen-basis projection with PyramidKV layer ranks for seq_len > warmup_len.

    In batch/teacher-forcing mode the hook always receives the full sequence at once,
    so both phases use a single frozen_cold_start call — the difference is:
      - short (T <= warmup_len): SVD computed on all T tokens → optimal rank-R basis
      - long  (T >  warmup_len): SVD on first warmup_len tokens, rest projected
    The warmup phase uses a flat warmup_rank; frozen phase uses layer-adaptive ranks.
    """
    def _hybrid_attn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        _, _, seq_len, _ = key.shape
        device = query.device

        # Choose rank: flat during warmup window, layer-adaptive beyond it
        if seq_len <= warmup_len:
            rank = warmup_rank
        else:
            rank = pyramid_ranks[min(layer_idx, len(pyramid_ranks) - 1)]

        out_heads = []
        for b in range(key.shape[0]):
            out_batch = []
            for h in range(key.shape[1]):
                k_h = torch_to_jax(key[b, h])
                v_h = torch_to_jax(value[b, h])
                q_h = torch_to_jax(query[b, h])

                state = frozen_cold_start(k_h, v_h, rank, warmup_len=warmup_len)
                out_h = frozen_compressed_attention(q_h, state, causal=True)
                out_batch.append(jax_to_torch(out_h, device))

            out_heads.append(torch.stack(out_batch, dim=0))

        return torch.stack(out_heads, dim=0).transpose(1, 2), None

    return _hybrid_attn


def _make_sinked_attn_fn(sink_len, window_len, warmup_len, pyramid_ranks):
    """
    Full-precision sink + recent tokens, frozen-basis compressed middle.
    Per-layer rank budget from pyramid_ranks (PyramidKV-style).
    Sink concept: Xiao et al. (2023) arXiv:2309.17453
    """
    def _sinked_attn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        rank = pyramid_ranks[min(layer_idx, len(pyramid_ranks) - 1)]
        device = query.device

        out_heads = []
        for b in range(key.shape[0]):
            out_batch = []
            for h in range(key.shape[1]):
                k_h = torch_to_jax(key[b, h])
                v_h = torch_to_jax(value[b, h])
                q_h = torch_to_jax(query[b, h])

                state = sinked_cold_start(k_h, v_h, rank,
                                          sink_len=sink_len,
                                          window_len=window_len,
                                          warmup_len=warmup_len)
                out_h = sinked_attention(q_h, state, causal=True)
                out_batch.append(jax_to_torch(out_h, device))

            out_heads.append(torch.stack(out_batch, dim=0))

        return torch.stack(out_heads, dim=0).transpose(1, 2), None

    return _sinked_attn


def patch_gpt2_sinked(
    model,
    sink_len: int = 64,
    window_len: int = 64,
    warmup_len: int = 512,
    pyramid_ranks=None,
) -> str:
    """
    Registers sinked attention and activates it for model.
    First/last sink_len/window_len tokens kept full-precision per StreamingLLM.
    Middle tokens compressed with frozen-basis SVD at PyramidKV layer ranks.
    Returns original _attn_implementation for restoration.
    """
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS

    if pyramid_ranks is None:
        pyramid_ranks = PYRAMID_RANKS

    ALL_ATTENTION_FUNCTIONS.register(
        "sinked_pyramid",
        _make_sinked_attn_fn(sink_len, window_len, warmup_len, pyramid_ranks),
    )
    original_impl = model.config._attn_implementation
    model.config._attn_implementation = "sinked_pyramid"
    return original_impl


def patch_gpt2_hybrid(
    model,
    warmup_len: int = 512,
    warmup_rank: int = 16,
    pyramid_ranks=None,
) -> str:
    """
    Registers the hybrid attention and activates it for model.
    Short sequences use Brand's-equivalent full SVD at warmup_rank.
    Long sequences switch to PyramidKV frozen-basis at layer-adaptive ranks.
    Returns the original _attn_implementation string for restoration.
    """
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS

    if pyramid_ranks is None:
        pyramid_ranks = PYRAMID_RANKS

    ALL_ATTENTION_FUNCTIONS.register(
        "hybrid_pyramid",
        _make_hybrid_pyramid_fn(warmup_len, warmup_rank, pyramid_ranks),
    )
    original_impl = model.config._attn_implementation
    model.config._attn_implementation = "hybrid_pyramid"
    return original_impl


def patch_gpt2_frozen_basis(model, pyramid_ranks=None) -> str:
    """
    Registers the frozen-basis pyramid attention and activates it for model.
    Returns the original _attn_implementation string so it can be restored.
    """
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS

    if pyramid_ranks is None:
        pyramid_ranks = PYRAMID_RANKS

    ALL_ATTENTION_FUNCTIONS.register(
        "frozen_pyramid", _make_frozen_pyramid_fn(pyramid_ranks)
    )
    original_impl = model.config._attn_implementation
    model.config._attn_implementation = "frozen_pyramid"
    return original_impl


def patch_gpt2_for_compression(
    model, phi: jnp.ndarray, rank: int, T_max: int
) -> str:
    """
    Registers the SVD-compressed attention function and activates it for model.
    Returns the original _attn_implementation string so it can be restored.
    """
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        "svd_compressed", _make_compressed_attn_fn(phi, rank, T_max)
    )
    original_impl = model.config._attn_implementation
    model.config._attn_implementation = "svd_compressed"
    return original_impl


def _make_sketch_attn_fn(rank: int, T_max: int, sketch_depth: int = 4):
    """
    Attention hook using sketch_cold_start: builds a count-min sketch from each
    head's key vectors, derives importance weights via query(), then calls
    sketch_cold_start so the same Phi drives both the sketch and the Halko SVD.
    """
    from loop import JaxTensorSketchStore, sketch_cold_start

    def _sketch_attn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch, n_heads, _, head_dim = key.shape
        device = query.device

        out_heads = []
        for b in range(batch):
            out_batch = []
            for h in range(n_heads):
                k_h = torch_to_jax(key[b, h])
                v_h = torch_to_jax(value[b, h])
                q_h = torch_to_jax(query[b, h])

                store = JaxTensorSketchStore(
                    depth=sketch_depth, width=max(rank, 1), embedding_dim=head_dim
                )
                sketch_state = store.init_state()
                for i in range(k_h.shape[0]):
                    sketch_state = JaxTensorSketchStore.inner_loop_update(
                        sketch_state, k_h[i], store.d, store.w
                    )

                state = sketch_cold_start(k_h, v_h, store, sketch_state, rank, T_max)
                out_h, _ = compressed_attention_with_weights(q_h, state, T_max, causal=True)
                out_batch.append(jax_to_torch(out_h, device))

            out_heads.append(torch.stack(out_batch, dim=0))

        return torch.stack(out_heads, dim=0).transpose(1, 2), None

    return _sketch_attn


def patch_gpt2_sketch(
    model, rank: int, T_max: int, sketch_depth: int = 4
) -> str:
    """
    Registers sketch-driven compressed attention and activates it for model.
    Uses the same Phi matrix for both count-min importance scoring and the
    Halko randomized SVD — the core paper contribution.
    Returns original _attn_implementation for restoration.
    """
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        "sketch_compressed", _make_sketch_attn_fn(rank, T_max, sketch_depth)
    )
    original_impl = model.config._attn_implementation
    model.config._attn_implementation = "sketch_compressed"
    return original_impl


def unpatch_gpt2(model, original_impl: str) -> None:
    model.config._attn_implementation = original_impl


# ---------------------------------------------------------------------------
# Generic GQA-aware patches — work for LLaMA, Mistral, TinyLlama, etc.
# Handles Grouped Query Attention: one KV state per KV head, each serving
# (n_q_heads // n_kv_heads) query heads.
# ---------------------------------------------------------------------------

def _make_generic_sinked_fn(sink_len, window_len, warmup_len, pyramid_ranks):
    """
    GQA-aware sinked attention hook for any model using ALL_ATTENTION_FUNCTIONS.
    K/V are compressed once per KV head; each Q head in the group attends to that state.
    """
    def _attn(
        module,
        query: torch.Tensor,   # (batch, n_q_heads, seq_len, head_dim)
        key: torch.Tensor,     # (batch, n_kv_heads, seq_len, head_dim)
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        rank = pyramid_ranks[min(layer_idx, len(pyramid_ranks) - 1)]
        batch, n_kv_heads, _, _ = key.shape
        n_q_heads = query.shape[1]
        groups = n_q_heads // n_kv_heads  # =1 for MHA, >1 for GQA
        device = query.device
        dtype = query.dtype  # preserve model dtype (bfloat16, float32, …)

        out_heads = []
        for b in range(batch):
            out_b = []
            for kv_h in range(n_kv_heads):
                k_h = torch_to_jax(key[b, kv_h])
                v_h = torch_to_jax(value[b, kv_h])
                state = sinked_cold_start(k_h, v_h, rank,
                                          sink_len=sink_len,
                                          window_len=window_len,
                                          warmup_len=warmup_len)
                for g in range(groups):
                    q_h = torch_to_jax(query[b, kv_h * groups + g])
                    out_h = sinked_attention(q_h, state, causal=True)
                    out_b.append(jax_to_torch(out_h, device).to(dtype))
            out_heads.append(torch.stack(out_b, dim=0))  # (n_q_heads, seq, head_dim)

        return torch.stack(out_heads, dim=0).transpose(1, 2), None  # (batch, seq, n_q_heads, head_dim)

    return _attn


def _oja_jit(K: jnp.ndarray, V: jnp.ndarray, rank: int, eta: float = 0.1):
    """
    Fast JAX-scan Oja rule: streams K/V tokens and returns stale projections.

    Stale projections model the published OjaKV algorithm — each token's
    coordinate is computed against the basis AT THAT TOKEN'S TIME, not the
    final basis. Old tokens have drifted, which is OjaKV's core error source.

    Returns:
      basis_k, basis_v : (rank, d)  — final basis (right singular vectors)
      stale_proj_k     : (T, rank)  — each token's projection when it arrived
      stale_proj_v     : (T, rank)
    """
    _, d = K.shape
    key_init = jax.random.PRNGKey(0)
    V_k = jax.random.normal(key_init, (rank, d)) * (d ** -0.5)
    V_v = jax.random.normal(jax.random.PRNGKey(1), (rank, d)) * (d ** -0.5)

    def oja_step(carry, inputs):
        Vk, Vv = carry
        kt, vt = inputs
        # project with current basis (STALE for previously stored tokens)
        pk = kt @ Vk.T    # (rank,)
        pv = vt @ Vv.T    # (rank,)
        # Oja rank-1 update
        Vk = Vk + eta * jnp.outer(pk, kt - pk @ Vk) / (jnp.dot(kt, kt) + 1e-9)
        Vv = Vv + eta * jnp.outer(pv, vt - pv @ Vv) / (jnp.dot(vt, vt) + 1e-9)
        Vk = Vk / (jnp.linalg.norm(Vk, axis=1, keepdims=True) + 1e-9)
        Vv = Vv / (jnp.linalg.norm(Vv, axis=1, keepdims=True) + 1e-9)
        return (Vk, Vv), (pk, pv)

    (basis_k, basis_v), (stale_k, stale_v) = jax.lax.scan(
        oja_step, (V_k, V_v), (K, V)
    )
    return basis_k, basis_v, stale_k, stale_v


_oja_jit_compiled = jax.jit(_oja_jit, static_argnums=(2,))


def _make_generic_oja_fn(pyramid_ranks):
    """
    GQA-aware OjaKV baseline hook using JAX-scan Oja rule (JIT-compiled, fast).
    Stale projections model the published OjaKV drift at long context.
    """
    def _attn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        rank = pyramid_ranks[min(layer_idx, len(pyramid_ranks) - 1)]
        batch, n_kv_heads, seq_len, head_dim = key.shape
        n_q_heads = query.shape[1]
        groups = n_q_heads // n_kv_heads
        device = query.device
        dtype = query.dtype
        scale = head_dim ** -0.5

        out_heads = []
        for b in range(batch):
            out_b = []
            for kv_h in range(n_kv_heads):
                k_h = torch_to_jax(key[b, kv_h])   # (T, d)
                v_h = torch_to_jax(value[b, kv_h])
                basis_k, basis_v, stale_k, stale_v = _oja_jit_compiled(k_h, v_h, rank)
                # Attention: query projected into FINAL basis, keys are STALE projections
                mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
                for g in range(groups):
                    q_h = torch_to_jax(query[b, kv_h * groups + g])  # (T, d)
                    q_proj = q_h @ basis_k.T                           # (T, rank)
                    scores = (q_proj @ stale_k.T) * scale              # (T, T)
                    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
                    weights = jax.nn.softmax(scores, axis=-1)          # (T, T)
                    out_h = (weights @ stale_v) @ basis_v              # (T, d)
                    out_b.append(jax_to_torch(out_h, device).to(dtype))
            out_heads.append(torch.stack(out_b, dim=0))

        return torch.stack(out_heads, dim=0).transpose(1, 2), None

    return _attn


def patch_model_sinked(
    model,
    sink_len: int = 64,
    window_len: int = 64,
    warmup_len: int = 512,
    pyramid_ranks=None,
    key: str = "sinked_gqa",
) -> str:
    """
    Generic sinked attention patch for any model (GPT-2, LLaMA, Mistral, TinyLlama…).
    Registers the hook globally and sets model.config._attn_implementation.
    Returns original implementation string for restoration.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if pyramid_ranks is None:
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        pyramid_ranks = _default_pyramid(n_layers)
    ALL_ATTENTION_FUNCTIONS.register(
        key, _make_generic_sinked_fn(sink_len, window_len, warmup_len, pyramid_ranks)
    )
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original


def patch_model_oja(
    model,
    pyramid_ranks=None,
    key: str = "oja_gqa",
) -> str:
    """
    Generic OjaKV baseline patch (bare Oja rule, no sink/window, no frozen transition).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if pyramid_ranks is None:
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        pyramid_ranks = _default_pyramid(n_layers)
    ALL_ATTENTION_FUNCTIONS.register(
        key, _make_generic_oja_fn(pyramid_ranks)
    )
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original


def unpatch_model(model, original_impl: str) -> None:
    model.config._attn_implementation = original_impl


def _make_generic_frozen_flat_fn(rank: int):
    """
    GQA-aware flat-rank frozen-basis hook — pure PyTorch, no JAX conversions.

    Avoids the 100x slowdown from per-head torch→jax→torch round-trips by
    batching SVD and attention across all KV heads in one shot.

    Per forward call:
      1. SVD of K across all KV heads: (batch, n_kv, T, d) → basis (batch, n_kv, r, d)
      2. Project K, V onto basis: (batch, n_kv, T, r)
      3. Causal attention for each Q head using its group's compressed K/V
    """
    def _attn(
        module,
        query: torch.Tensor,   # (batch, n_q_heads, seq_len, head_dim)
        key: torch.Tensor,     # (batch, n_kv_heads, seq_len, head_dim)
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch, n_kv_heads, seq_len, head_dim = key.shape
        n_q_heads = query.shape[1]
        groups = n_q_heads // n_kv_heads
        device, dtype = query.device, query.dtype
        r = min(rank, seq_len, head_dim)
        scale = head_dim ** -0.5

        # --- Batched SVD over all KV heads ---
        # torch.linalg.svd on (batch*n_kv, T, d) → Vt: (batch*n_kv, d, d)
        k_flat = key.reshape(batch * n_kv_heads, seq_len, head_dim).float()
        v_flat = value.reshape(batch * n_kv_heads, seq_len, head_dim).float()
        _, _, Vt_k = torch.linalg.svd(k_flat, full_matrices=False)  # (B*nkv, min(T,d), d)
        _, _, Vt_v = torch.linalg.svd(v_flat, full_matrices=False)
        basis_k = Vt_k[:, :r, :]   # (B*nkv, r, d)
        basis_v = Vt_v[:, :r, :]

        # Project K, V onto their bases: (B*nkv, T, r)
        proj_k = torch.bmm(k_flat, basis_k.transpose(1, 2))
        proj_v = torch.bmm(v_flat, basis_v.transpose(1, 2))

        # Reshape back: (batch, n_kv, T, r) and (batch, n_kv, r, d)
        proj_k  = proj_k.reshape(batch, n_kv_heads, seq_len, r)
        proj_v  = proj_v.reshape(batch, n_kv_heads, seq_len, r)
        basis_k = basis_k.reshape(batch, n_kv_heads, r, head_dim)
        basis_v = basis_v.reshape(batch, n_kv_heads, r, head_dim)

        # --- Causal mask (batch, 1, T, T) for broadcasting ---
        causal_mask = torch.tril(torch.ones(seq_len, seq_len,
                                            device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

        # --- Vectorised over Q heads per KV group (eliminates the inner loop) ---
        # query: (batch, n_q_heads, T, d) → grouped by KV head
        # Expand basis/proj to match Q-head grouping: (batch, n_kv, r, d) → (batch, n_q, r, d)
        basis_k_exp = basis_k.repeat_interleave(groups, dim=1)  # (batch, n_q, r, d)
        basis_v_exp = basis_v.repeat_interleave(groups, dim=1)
        proj_k_exp  = proj_k.repeat_interleave(groups, dim=1)   # (batch, n_q, T, r)
        proj_v_exp  = proj_v.repeat_interleave(groups, dim=1)

        q_f = query.float()                                       # (batch, n_q, T, d)
        q_proj = torch.einsum("bhtd,bhrd->bhtr", q_f, basis_k_exp)        # (batch, n_q, T, r)
        scores = torch.einsum("bhtr,bhsr->bhts", q_proj, proj_k_exp) * scale  # (batch, n_q, T, T)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                            # (batch, n_q, T, T)
        out = torch.einsum("bhts,bhsr->bhtr", weights, proj_v_exp)        # (batch, n_q, T, r)
        out = torch.einsum("bhtr,bhrd->bhtd", out, basis_v_exp).to(dtype) # (batch, n_q, T, d)

        # (batch, n_q, T, d) → transpose → (batch, T, n_q, d)
        return out.transpose(1, 2), None

    return _attn


def patch_model_frozen_flat(model, rank: int, key: str = "frozen_flat") -> str:
    """
    Registers flat-rank frozen-basis attention and activates it for model.
    Identical rank budget for every layer — used for rank sweep benchmarks.
    Returns original _attn_implementation for restoration.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(key, _make_generic_frozen_flat_fn(rank))
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original


def _default_pyramid(n_layers: int) -> List[int]:
    """Divide layers into 4 equal quartiles with ranks [32,16,8,4]."""
    q = n_layers // 4
    r = n_layers % 4
    ranks = [32] * (q + (1 if r > 0 else 0))
    ranks += [16] * (q + (1 if r > 1 else 0))
    ranks += [8]  * (q + (1 if r > 2 else 0))
    ranks += [4]  * q
    return ranks


def load_model(model_name: str, device: str = "cpu", dtype=torch.bfloat16):
    """Load any CausalLM + tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, use_cache=False
    ).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    device: str = "cpu",
) -> float:
    """
    Teacher-forcing perplexity over a list of strings.
    Call patch_gpt2_for_compression before this for the compressed variant.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids, labels=input_ids, use_cache=False)
        nll = outputs.loss.item()
        n_tokens = input_ids.shape[1] - 1
        total_nll += nll * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def load_gpt2(device: str = "cpu"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return model, tokenizer


def make_phi_for_gpt2(rank: int, head_dim: int = 64, seed: int = 42) -> jnp.ndarray:
    """He-like random sketch matching JaxTensorSketchStore's Phi_init."""
    key = jax.random.PRNGKey(seed)
    stddev = jnp.sqrt(2.0 / head_dim)
    return jax.random.normal(key, (rank, head_dim)) * stddev


# --- Smoke test ---
if __name__ == "__main__":
    RANK = 16
    T_MAX = 512
    HEAD_DIM = 64
    MAX_LEN = 64

    print("Loading GPT-2...")
    model, tokenizer = load_gpt2()
    phi = make_phi_for_gpt2(RANK, HEAD_DIM)

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers have revolutionized natural language processing.",
    ]

    ppl_exact = compute_perplexity(model, tokenizer, sample_texts, max_length=MAX_LEN)
    print(f"Exact perplexity:               {ppl_exact:.2f}")

    original_impl = patch_gpt2_for_compression(model, phi, RANK, T_MAX)
    ppl_compressed = compute_perplexity(model, tokenizer, sample_texts, max_length=MAX_LEN)
    unpatch_gpt2(model, original_impl)
    print(f"Compressed (rank={RANK}) perplexity:  {ppl_compressed:.2f}")
    print(f"Perplexity ratio:               {ppl_compressed / ppl_exact:.3f}x")
    print("model_hooks.py: smoke test complete")
