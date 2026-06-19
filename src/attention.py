import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from kv_state import CompressedKVState, TKVState, SinkedKVState, reconstruct_kv, update_importance


def compressed_attention(
    q: jnp.ndarray,              # (n_q, d)
    state: CompressedKVState,
    T_max: int,
    scale: Optional[float] = None,
    causal: bool = False,
) -> jnp.ndarray:
    """
    Attention output without reconstructing K or V.

    Uses the SVD identity:
      scores = q K^T = q (U_k sigma_k V_k^T)^T
             = (q @ V_k.T) * sigma_k @ U_k.T
      output = softmax(scores) @ (U_v * sigma_v) @ V_v

    causal=True: query at row i attends only to keys at positions <= i (for
    full-sequence teacher-forcing). causal=False: all queries attend to all
    positions < seq_len (for autoregressive single-token decoding).
    """
    n_q, d = q.shape
    if scale is None:
        scale = d ** -0.5

    # Score computation in compressed form: (n_q, T_max)
    q_proj = q @ state.V_k.T                          # (n_q, R_max)
    scores = (q_proj * state.sigma_k) @ state.U_k.T   # (n_q, T_max)
    scores = scores * scale

    pos = jnp.arange(T_max)                           # (T_max,)
    if causal:
        # query i attends to key j only if j <= i AND j < seq_len
        query_idx = jnp.arange(n_q)[:, None]          # (n_q, 1)
        key_idx = pos[None, :]                         # (1, T_max)
        valid = (key_idx <= query_idx) & (key_idx < state.seq_len)
    else:
        valid = pos[None, :] < state.seq_len           # (1, T_max) broadcast

    scores = jnp.where(valid, scores, jnp.finfo(jnp.float32).min)

    weights = jax.nn.softmax(scores, axis=-1)          # (n_q, T_max)

    # Output in compressed form: (n_q, d)
    w_proj = weights @ state.U_v                       # (n_q, R_max)
    out = (w_proj * state.sigma_v) @ state.V_v         # (n_q, d)

    return out


def compressed_attention_with_weights(
    q: jnp.ndarray,              # (n_q, d)
    state: CompressedKVState,
    T_max: int,
    scale: Optional[float] = None,
    causal: bool = False,
    alpha: float = 0.9,
) -> Tuple[jnp.ndarray, CompressedKVState]:
    """
    Like compressed_attention, but also runs the Moment-KV EMA update.

    Returns (output, updated_state) where updated_state has refreshed
    importance scores: I_i = alpha * I_i + mean_q(attn_weights[q, i]).
    The mean over queries summarises which positions were attended to most,
    matching Moment-KV's head-averaged importance signal.

    Call this instead of compressed_attention during autoregressive decoding
    to keep importance scores current for rank adaptation.
    """
    n_q, d = q.shape
    if scale is None:
        scale = d ** -0.5

    q_proj = q @ state.V_k.T
    scores = (q_proj * state.sigma_k) @ state.U_k.T * scale

    pos = jnp.arange(T_max)
    if causal:
        valid = (pos[None, :] <= jnp.arange(n_q)[:, None]) & (pos[None, :] < state.seq_len)
    else:
        valid = pos[None, :] < state.seq_len

    scores = jnp.where(valid, scores, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(scores, axis=-1)          # (n_q, T_max)

    w_proj = weights @ state.U_v
    out = (w_proj * state.sigma_v) @ state.V_v         # (n_q, d)

    # Moment-KV EMA: aggregate attention over queries then update state
    mean_weights = weights.mean(axis=0)                # (T_max,)
    new_state = update_importance(state, mean_weights, alpha=alpha)

    return out, new_state


def tkv_compressed_attention(
    q: jnp.ndarray,           # (n_q, d)
    state: TKVState,
    scale: Optional[float] = None,
    causal: bool = False,
) -> jnp.ndarray:
    """
    Attention in the TKV frozen-phase low-rank basis. No K/V reconstruction needed.

    Exact in the projected subspace:
      scores[q, t] = (q @ basis_k.T) · proj_k[t]   (inner product in R-dim space)
      output        = softmax(scores) @ proj_v @ basis_v

    The approximation is that keys/values outside the basis subspace are ignored,
    which is controlled by the rank and warmup quality. Scale uses head_dim d
    (not rank R) to preserve the original attention temperature.
    """
    n_q, d = q.shape
    T = state.seq_len
    if scale is None:
        scale = d ** -0.5

    proj_k = jnp.array(state.proj_k)  # (T, R)
    proj_v = jnp.array(state.proj_v)  # (T, R)

    q_proj = q @ state.basis_k.T                   # (n_q, R)
    scores = (q_proj @ proj_k.T) * scale           # (n_q, T)

    if causal:
        query_idx = jnp.arange(n_q)[:, None]       # (n_q, 1)
        key_idx   = jnp.arange(T)[None, :]         # (1, T)
        scores = jnp.where(key_idx <= query_idx, scores, jnp.finfo(jnp.float32).min)

    weights = jax.nn.softmax(scores, axis=-1)      # (n_q, T)
    out = (weights @ proj_v) @ state.basis_v       # (n_q, d)
    return out


def sinked_attention(
    q: jnp.ndarray,         # (n_q, d)
    state: SinkedKVState,
    scale: Optional[float] = None,
    causal: bool = False,
) -> jnp.ndarray:
    """
    Attention over three segments with a single softmax:
      sink   — full precision (always exact)
      middle — TKV (frozen-basis) compressed  (may be None when T is short)
      recent — full precision (always exact)

    Absolute positions are used for the causal mask so each query sees
    exactly the right subset of sink / middle / recent tokens.

    Cite: StreamingLLM sink concept — Xiao et al. (2023) arXiv:2309.17453
    """
    n_q, d = q.shape
    if scale is None:
        scale = d ** -0.5
    neg_inf = jnp.finfo(jnp.float32).min

    n_sink  = state.sink_len
    T_mid   = state.middle.seq_len if state.middle is not None else 0
    n_recent = len(state.recent_k)

    # --- sink (full precision) ---
    sink_k = jnp.array(state.sink_k)   # (n_sink, d)
    sink_v = jnp.array(state.sink_v)
    sink_scores = (q @ sink_k.T) * scale    # (n_q, n_sink)

    # --- recent (full precision) ---
    recent_k = jnp.array(state.recent_k)   # (n_recent, d)
    recent_v = jnp.array(state.recent_v)
    recent_scores = (q @ recent_k.T) * scale  # (n_q, n_recent)

    if causal:
        q_abs = jnp.arange(n_q)[:, None]                         # (n_q, 1)
        sink_pos   = jnp.arange(n_sink)[None, :]                 # (1, n_sink)
        rec_pos    = (jnp.arange(n_recent) + n_sink + T_mid)[None, :]
        sink_scores   = jnp.where(sink_pos <= q_abs, sink_scores,   neg_inf)
        recent_scores = jnp.where(rec_pos  <= q_abs, recent_scores, neg_inf)

    score_parts = [sink_scores]

    # --- middle (compressed) ---
    if T_mid > 0:
        proj_k  = jnp.array(state.middle.proj_k)    # (T_mid, R)
        proj_v  = jnp.array(state.middle.proj_v)    # (T_mid, R)
        q_proj  = q @ state.middle.basis_k.T         # (n_q, R)
        mid_scores = (q_proj @ proj_k.T) * scale    # (n_q, T_mid)
        if causal:
            mid_pos = (jnp.arange(T_mid) + n_sink)[None, :]
            mid_scores = jnp.where(mid_pos <= q_abs, mid_scores, neg_inf)
        score_parts.append(mid_scores)

    score_parts.append(recent_scores)

    # single softmax over [sink | middle | recent]
    weights = jax.nn.softmax(jnp.concatenate(score_parts, axis=-1), axis=-1)  # (n_q, total_T)

    sink_w = weights[:, :n_sink]
    out = sink_w @ sink_v                                         # (n_q, d)

    if T_mid > 0:
        mid_w = weights[:, n_sink : n_sink + T_mid]
        out   = out + (mid_w @ proj_v) @ state.middle.basis_v
        rec_w = weights[:, n_sink + T_mid:]
    else:
        rec_w = weights[:, n_sink:]

    out = out + rec_w @ recent_v
    return out


def exact_attention_from_state(
    q: jnp.ndarray,              # (n_q, d)
    state: CompressedKVState,
    T_max: int,
    scale: Optional[float] = None,
) -> jnp.ndarray:
    """
    Exact attention using reconstructed K, V. For benchmarking only.
    """
    d = q.shape[-1]
    if scale is None:
        scale = d ** -0.5

    K, V = reconstruct_kv(state, T_max)  # (T_max, d)

    scores = (q @ K.T) * scale           # (n_q, T_max)
    pos = jnp.arange(T_max)
    scores = jnp.where(pos[None, :] < state.seq_len, scores, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ V                   # (n_q, d)


# --- Unit test ---
if __name__ == "__main__":
    from kv_state import init_compressed_kv, append_token

    T_MAX, R_MAX, D, N_Q = 256, 16, 64, 4
    key = jax.random.PRNGKey(42)

    # Build a KV cache with fast singular value decay for good compression
    k1, k2, k3, k4 = jax.random.split(key, 4)
    base = jax.random.normal(k1, (4, D))  # low-rank base
    K_true = (jax.random.normal(k2, (64, 4)) @ base) * 0.1
    V_true = jax.random.normal(k3, (64, D)) * 0.1

    state = init_compressed_kv(T_MAX, R_MAX, D)
    for i in range(64):
        state = append_token(state, K_true[i], V_true[i], R_MAX)

    Q = jax.random.normal(k4, (N_Q, D)) * 0.1

    out_compressed = compressed_attention(Q, state, T_MAX)
    out_exact = exact_attention_from_state(Q, state, T_MAX)

    rel_err = jnp.linalg.norm(out_compressed - out_exact) / (jnp.linalg.norm(out_exact) + 1e-9)
    print(f"Attention output rel error: {rel_err:.4f}")
    assert rel_err < 0.1, f"Attention error too large: {rel_err}"
    print("attention.py: all checks passed")
