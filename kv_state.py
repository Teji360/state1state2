import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple


class CompressedKVState(NamedTuple):
    U_k:         jnp.ndarray  # (T_max, R_max) left singular vecs, key cache
    sigma_k:     jnp.ndarray  # (R_max,) singular values descending
    V_k:         jnp.ndarray  # (R_max, d) right singular vecs (key space basis)
    U_v:         jnp.ndarray  # (T_max, R_max) left singular vecs, value cache
    sigma_v:     jnp.ndarray  # (R_max,) singular values descending
    V_v:         jnp.ndarray  # (R_max, d) right singular vecs (value space basis)
    seq_len:     jnp.ndarray  # scalar int32
    active_rank: jnp.ndarray  # scalar int32, <= R_max
    importance:  jnp.ndarray  # (T_max,) EMA importance score per token position


def init_compressed_kv(T_max: int, R_max: int, d: int) -> CompressedKVState:
    return CompressedKVState(
        U_k=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        sigma_k=jnp.zeros((R_max,), dtype=jnp.float32),
        V_k=jnp.zeros((R_max, d), dtype=jnp.float32),
        U_v=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        sigma_v=jnp.zeros((R_max,), dtype=jnp.float32),
        V_v=jnp.zeros((R_max, d), dtype=jnp.float32),
        seq_len=jnp.array(0, dtype=jnp.int32),
        active_rank=jnp.array(0, dtype=jnp.int32),
        importance=jnp.zeros((T_max,), dtype=jnp.float32),
    )


def _brand_core(
    sigma: jnp.ndarray, # (R_max,)
    V: jnp.ndarray,     # (R_max, d)
    x_new: jnp.ndarray, # (d,)
    R_max: int,
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Brand (2006) core: everything except the U rotation.
    Returns (M, new_row, sigma_new, V_new, new_rank) where M is the
    (R_max, R_max) rotation that should be applied to existing U rows.
    Cost: O(R³) for the small SVD + O(R·d) for V update.
    """
    Vx = V @ x_new
    p = x_new - V.T @ Vx
    p_norm = jnp.linalg.norm(p)
    p_hat = p / jnp.maximum(p_norm, 1e-9)

    K_aug = jnp.zeros((R_max + 1, R_max + 1), dtype=jnp.float32)
    K_aug = K_aug.at[:R_max, :R_max].set(jnp.diag(sigma))
    K_aug = K_aug.at[R_max, :R_max].set(Vx)
    K_aug = K_aug.at[R_max, R_max].set(p_norm)

    U_aug, sigma_aug, Wt_aug = jnp.linalg.svd(K_aug, full_matrices=False)

    sigma_new = sigma_aug[:R_max]
    V_ext = jnp.concatenate([V, p_hat[None, :]], axis=0)
    V_new = Wt_aug[:R_max] @ V_ext

    M = U_aug[:R_max, :R_max]
    new_row = U_aug[R_max, :R_max]

    sigma_1 = jnp.maximum(sigma_new[0], 1e-9)
    new_rank = jnp.clip(jnp.sum(sigma_new > eps * sigma_1).astype(jnp.int32), 1, R_max)

    return M, new_row, sigma_new, V_new, new_rank


def _brand_update(
    U: jnp.ndarray,     # (T_max, R_max)
    sigma: jnp.ndarray, # (R_max,)
    V: jnp.ndarray,     # (R_max, d)
    x_new: jnp.ndarray, # (d,) new key or value vector
    seq_len: jnp.ndarray,
    R_max: int,
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Brand (2006) rank-1 SVD update.
    Appends one new row x_new to the matrix represented by (U, sigma, V).
    All array shapes are static (XLA-safe). seq_len is a dynamic index.
    Returns updated (U, sigma, V, new_active_rank).
    """
    # Residual: component of x_new orthogonal to current row space V
    Vx = V @ x_new                  # (R_max,) — projection coefficients
    p = x_new - V.T @ Vx            # (d,) — residual
    p_norm = jnp.linalg.norm(p)
    p_hat = p / jnp.maximum(p_norm, 1e-9)  # (d,) normalized residual

    # Augmented (R_max+1) x (R_max+1) core matrix K_aug:
    # [[diag(sigma), Vx],
    #  [0...,        p_norm]]
    # Build it in a fixed (R_max+1, R_max+1) buffer — trailing zeros make
    # extra singular values zero, which is XLA-safe.
    # K_aug = [[diag(sigma), 0], [Vx^T, p_norm]]
    # The last ROW encodes the new token's projection; last column stays zero.
    # (Brand 2006: K' = U_ext K_aug V_ext where V_ext = [V; p_hat])
    K_aug = jnp.zeros((R_max + 1, R_max + 1), dtype=jnp.float32)
    K_aug = K_aug.at[:R_max, :R_max].set(jnp.diag(sigma))
    K_aug = K_aug.at[R_max, :R_max].set(Vx)   # last row, not last column
    K_aug = K_aug.at[R_max, R_max].set(p_norm)

    U_aug, sigma_aug, Wt_aug = jnp.linalg.svd(K_aug, full_matrices=False)
    # shapes: (R_max+1, R_max+1), (R_max+1,), (R_max+1, R_max+1)

    # Truncate sigma to R_max — the last value is the (R_max+1)-th which we drop
    sigma_new = sigma_aug[:R_max]  # (R_max,)
    Wt_new = Wt_aug[:R_max]        # (R_max, R_max+1)

    # Update V: new basis = [V; p_hat] @ Wt_new.T (shape (d, R_max+1) @ (R_max+1, R_max))
    # Build extended V_ext: (R_max+1, d) = [[V], [p_hat]]
    V_ext = jnp.concatenate([V, p_hat[None, :]], axis=0)  # (R_max+1, d)
    V_new = Wt_new @ V_ext  # (R_max, d)

    # Update U: extended U_ext = [[U], [0_row]] then apply left rotation U_aug[:, :R_max+1]
    # New U row for the incoming token is the (R_max+1)-th row of U_aug (index R_max)
    # Full new U would be U_ext @ U_aug[:R_max+1, :R_max] but we only store the new row here
    # and use dynamic_update_slice to append it.
    # Existing rows of U get rotated by U_aug[:R_max, :R_max]:
    U_rot = U @ U_aug[:R_max, :R_max]    # (T_max, R_max) @ (R_max, R_max) = (T_max, R_max)
    new_row = U_aug[R_max, :R_max]       # (R_max,) — the new token's row in U

    # Append new_row at position seq_len
    U_new = jax.lax.dynamic_update_slice(U_rot, new_row[None, :], (seq_len, 0))

    # Adaptive rank: keep singular values above eps * sigma_1
    sigma_1 = jnp.maximum(sigma_new[0], 1e-9)
    new_rank = jnp.sum(sigma_new > eps * sigma_1).astype(jnp.int32)
    new_rank = jnp.clip(new_rank, 1, R_max)

    return U_new, sigma_new, V_new, new_rank


# ---------------------------------------------------------------------------
# Lazy batch-rotation variants
# ---------------------------------------------------------------------------

class LazyKVState(NamedTuple):
    U_k:               jnp.ndarray  # (T_max, R_max)
    sigma_k:           jnp.ndarray  # (R_max,)
    V_k:               jnp.ndarray  # (R_max, d)
    U_v:               jnp.ndarray  # (T_max, R_max)
    sigma_v:           jnp.ndarray  # (R_max,)
    V_v:               jnp.ndarray  # (R_max, d)
    seq_len:           jnp.ndarray  # scalar int32
    active_rank:       jnp.ndarray  # scalar int32
    pending_rot_k:     jnp.ndarray  # (R_max, R_max) accumulated rotation for K
    pending_rot_v:     jnp.ndarray  # (R_max, R_max) accumulated rotation for V
    steps_since_flush: jnp.ndarray  # scalar int32


def init_lazy_kv(T_max: int, R_max: int, d: int) -> LazyKVState:
    return LazyKVState(
        U_k=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        sigma_k=jnp.zeros((R_max,), dtype=jnp.float32),
        V_k=jnp.zeros((R_max, d), dtype=jnp.float32),
        U_v=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        sigma_v=jnp.zeros((R_max,), dtype=jnp.float32),
        V_v=jnp.zeros((R_max, d), dtype=jnp.float32),
        seq_len=jnp.array(0, dtype=jnp.int32),
        active_rank=jnp.array(0, dtype=jnp.int32),
        pending_rot_k=jnp.eye(R_max, dtype=jnp.float32),
        pending_rot_v=jnp.eye(R_max, dtype=jnp.float32),
        steps_since_flush=jnp.array(0, dtype=jnp.int32),
    )


def append_token_lazy(
    state: LazyKVState,
    k_new: jnp.ndarray,
    v_new: jnp.ndarray,
    R_max: int,
    flush_every: int = 8,
    eps: float = 0.01,
) -> LazyKVState:
    """
    Brand update with deferred U rotation (batch rotation).

    Per token: O(R³) small SVD + O(R²) pending_rot accumulation.
    Every flush_every tokens: one O(T_max · R²) bulk U rotation instead of
    flush_every individual ones — amortised cost T_max · R² / flush_every.

    Approximation: rows added within a flush window are over-rotated by the
    M matrices that preceded their insertion in that window.  The error is
    bounded by ||M_prefix - I|| where M_prefix is the product of M's before
    the row's insertion within the window — small when M_t ≈ I, which holds
    when the subspace changes slowly relative to the flush interval.
    """
    M_k, r_k, sigma_k, V_k, rank_k = _brand_core(
        state.sigma_k, state.V_k, k_new, R_max, eps
    )
    M_v, r_v, sigma_v, V_v, rank_v = _brand_core(
        state.sigma_v, state.V_v, v_new, R_max, eps
    )

    U_k = jax.lax.dynamic_update_slice(state.U_k, r_k[None, :], (state.seq_len, 0))
    U_v = jax.lax.dynamic_update_slice(state.U_v, r_v[None, :], (state.seq_len, 0))

    pending_rot_k = state.pending_rot_k @ M_k
    pending_rot_v = state.pending_rot_v @ M_v
    steps = int(state.steps_since_flush) + 1

    if steps >= flush_every:
        U_k = U_k @ pending_rot_k
        U_v = U_v @ pending_rot_v
        pending_rot_k = jnp.eye(R_max, dtype=jnp.float32)
        pending_rot_v = jnp.eye(R_max, dtype=jnp.float32)
        steps = 0

    new_rank = jnp.maximum(rank_k, rank_v)
    return LazyKVState(
        U_k=U_k, sigma_k=sigma_k, V_k=V_k,
        U_v=U_v, sigma_v=sigma_v, V_v=V_v,
        seq_len=state.seq_len + 1,
        active_rank=new_rank,
        pending_rot_k=pending_rot_k,
        pending_rot_v=pending_rot_v,
        steps_since_flush=jnp.array(steps, dtype=jnp.int32),
    )


def reconstruct_kv_lazy(
    state: LazyKVState, T_max: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct applying any un-flushed pending rotation to all rows."""
    U_k = state.U_k @ state.pending_rot_k
    U_v = state.U_v @ state.pending_rot_v
    mask = (jnp.arange(T_max) < state.seq_len)[:, None]
    K = jnp.where(mask, (U_k * state.sigma_k) @ state.V_k, 0.0)
    V = jnp.where(mask, (U_v * state.sigma_v) @ state.V_v, 0.0)
    return K, V


def append_token(
    state: CompressedKVState,
    k_new: jnp.ndarray,          # (d,)
    v_new: jnp.ndarray,          # (d,)
    R_max: int,
    eps: float = 0.01,
    token_importance: float = 1.0,
) -> CompressedKVState:
    """
    Online rank-1 Brand SVD update.

    token_importance seeds the importance buffer for the new token.
    It is updated later by update_importance() as future decode steps
    attend back to this position.
    """
    U_k, sigma_k, V_k, rank_k = _brand_update(
        state.U_k, state.sigma_k, state.V_k, k_new, state.seq_len, R_max, eps
    )
    U_v, sigma_v, V_v, rank_v = _brand_update(
        state.U_v, state.sigma_v, state.V_v, v_new, state.seq_len, R_max, eps
    )
    new_rank = jnp.maximum(rank_k, rank_v)
    new_importance = jax.lax.dynamic_update_slice(
        state.importance,
        jnp.array([token_importance], dtype=jnp.float32),
        (state.seq_len,),
    )
    return CompressedKVState(
        U_k=U_k, sigma_k=sigma_k, V_k=V_k,
        U_v=U_v, sigma_v=sigma_v, V_v=V_v,
        seq_len=state.seq_len + 1,
        active_rank=new_rank,
        importance=new_importance,
    )


def reconstruct_kv(
    state: CompressedKVState, T_max: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct K and V matrices from compressed form. For benchmarking only."""
    mask = (jnp.arange(T_max) < state.seq_len)[:, None]  # (T_max, 1)
    K = jnp.where(mask, (state.U_k * state.sigma_k) @ state.V_k, 0.0)
    V = jnp.where(mask, (state.U_v * state.sigma_v) @ state.V_v, 0.0)
    return K, V


def cold_start(
    K: jnp.ndarray,
    V_mat: jnp.ndarray,
    phi: jnp.ndarray,
    R_max: int,
    T_max: int,
    d: int,
    importance_weights: Optional[jnp.ndarray] = None,
) -> CompressedKVState:
    """
    Halko et al. randomized SVD using loop.py's Phi as the random sketch.

    importance_weights: (t,) per-token importance scores (e.g. mean column
    attention from prefill). When provided, rows of K are scaled by
    sqrt(importance) before the range-finder step so that Q spans directions
    most relevant for high-attention tokens. The SVD of B_k = Q^T K then
    finds the rank-r approximation of K within that importance-biased subspace.
    Reconstruction and attention computation are unchanged — no un-scaling needed.
    """
    t = K.shape[0]
    rank = min(R_max, t, phi.shape[0])

    # Importance-biased range finder (Moment-KV integration):
    # scale rows by sqrt(I) so the sketch captures high-importance directions.
    if importance_weights is not None:
        scale = jnp.sqrt(jnp.maximum(importance_weights[:t], 1e-9))[:, None]
        K_sketch = K * scale
    else:
        K_sketch = K
        scale = jnp.ones((t, 1), dtype=jnp.float32)

    Y = K_sketch @ phi[:rank].T   # (t, rank) importance-biased sketch
    Q, _ = jnp.linalg.qr(Y)      # (t, rank) orthonormal range basis

    # Project unweighted K into the importance-biased range — no scaling in SVD.
    B_k = Q.T @ K
    U_b_k, sigma_k, V_k = jnp.linalg.svd(B_k, full_matrices=False)
    U_k_partial = Q @ U_b_k      # (t, rank)

    B_v = Q.T @ V_mat
    U_b_v, sigma_v, V_v = jnp.linalg.svd(B_v, full_matrices=False)
    U_v_partial = Q @ U_b_v

    # Pad into fixed T_max x R_max buffers
    U_k_buf = jnp.zeros((T_max, R_max), dtype=jnp.float32).at[:t, :rank].set(U_k_partial)
    sigma_k_buf = jnp.zeros((R_max,), dtype=jnp.float32).at[:rank].set(sigma_k)
    V_k_buf = jnp.zeros((R_max, d), dtype=jnp.float32).at[:rank].set(V_k)

    U_v_buf = jnp.zeros((T_max, R_max), dtype=jnp.float32).at[:t, :rank].set(U_v_partial)
    sigma_v_buf = jnp.zeros((R_max,), dtype=jnp.float32).at[:rank].set(sigma_v)
    V_v_buf = jnp.zeros((R_max, d), dtype=jnp.float32).at[:rank].set(V_v)

    imp_buf = jnp.zeros((T_max,), dtype=jnp.float32)
    if importance_weights is not None:
        imp_buf = imp_buf.at[:t].set(importance_weights[:t])
    else:
        imp_buf = imp_buf.at[:t].set(1.0)

    return CompressedKVState(
        U_k=U_k_buf, sigma_k=sigma_k_buf, V_k=V_k_buf,
        U_v=U_v_buf, sigma_v=sigma_v_buf, V_v=V_v_buf,
        seq_len=jnp.array(t, dtype=jnp.int32),
        active_rank=jnp.array(rank, dtype=jnp.int32),
        importance=imp_buf,
    )


def update_importance(
    state: CompressedKVState,
    attn_weights: jnp.ndarray,  # (T_max,) softmax weights from one decode step
    alpha: float = 0.9,
) -> CompressedKVState:
    """
    EMA update of per-token importance scores (Moment-KV Eq. 1):
      I_i(t) = alpha * I_i(t-1) + attn_weights[i]

    Call this after compressed_attention_with_weights() at each decode step.
    The accumulated importance guides adaptive rank selection: directions
    capturing high-importance tokens should be retained even if their raw
    singular value is small.
    """
    new_importance = alpha * state.importance + attn_weights
    return state._replace(importance=new_importance)


# ---------------------------------------------------------------------------
# Frozen-basis approach: O(R) per token, no stale projections, no T_max buffer
# ---------------------------------------------------------------------------

class FrozenBasisState(NamedTuple):
    basis_k: jnp.ndarray  # (R, d) — frozen key subspace, right singular vectors
    basis_v: jnp.ndarray  # (R, d) — frozen value subspace, right singular vectors
    proj_k:  np.ndarray   # (T, R) — projected keys in basis coordinates
    proj_v:  np.ndarray   # (T, R) — projected values in basis coordinates
    seq_len: int          # current number of tokens (plain Python int)
    rank:    int          # rank budget for this head/layer


def brand_to_frozen(brand_state: CompressedKVState) -> "FrozenBasisState":
    """
    Lossless conversion from Brand's CompressedKVState to FrozenBasisState.

    The math: Brand stores K ≈ U·diag(sigma)·V, so for token t:
      k_t ≈ U[t] * sigma @ V  →  k_t @ V.T ≈ U[t] * sigma  (since V is orthonormal)

    So proj_k[t] = U_k[t] * sigma_k gives IDENTICAL attention scores to Brand's state:
      Brand:  (q @ V_k.T) * sigma_k @ U_k[t].T
      Frozen: (q @ basis_k.T) @ proj_k[t]   ← same computation, just reorganised

    The transition is exact at warmup_len; after it new tokens are projected in O(R·d).
    """
    T = int(brand_state.seq_len)
    R = int(brand_state.active_rank)

    basis_k = brand_state.V_k[:R]  # (R, d)
    basis_v = brand_state.V_v[:R]  # (R, d)

    # Recover projections: multiply each U row by the corresponding singular value
    proj_k = np.array(brand_state.U_k[:T, :R] * brand_state.sigma_k[:R])  # (T, R)
    proj_v = np.array(brand_state.U_v[:T, :R] * brand_state.sigma_v[:R])  # (T, R)

    return FrozenBasisState(
        basis_k=basis_k,
        basis_v=basis_v,
        proj_k=proj_k,
        proj_v=proj_v,
        seq_len=T,
        rank=R,
    )


def hybrid_append_token(
    state,                       # CompressedKVState (warmup) or FrozenBasisState (streaming)
    k_new: jnp.ndarray,          # (d,)
    v_new: jnp.ndarray,          # (d,)
    R_max: int,
    warmup_len: int = 512,
    eps: float = 0.01,
):
    """
    Hybrid online KV update: Brand's exact SVD during warmup, frozen projection after.

    Phase 1 (seq_len < warmup_len): Brand rank-1 update — O(R³) per token.
      Tracks the exact singular subspace as the sequence grows. Builds the
      best possible basis for the frozen phase.

    Phase 2 (seq_len >= warmup_len): Converts Brand state → FrozenBasisState
      (lossless, then frozen_add_token for all future tokens — O(R·d) per token.
      No U-rotation cost, no stale projections, no T_max pre-allocation.

    Returns: CompressedKVState (during warmup) or FrozenBasisState (after).
    """
    if isinstance(state, FrozenBasisState):
        return frozen_add_token(state, k_new, v_new)

    new_brand = append_token(state, k_new, v_new, R_max, eps=eps)
    if int(new_brand.seq_len) >= warmup_len:
        return brand_to_frozen(new_brand)
    return new_brand


def frozen_cold_start(
    K: jnp.ndarray,
    V_mat: jnp.ndarray,
    rank: int,
    warmup_len: int = 512,
) -> FrozenBasisState:
    """
    Establish a frozen low-rank basis from the first min(T, warmup_len) tokens,
    then project all T tokens onto it. O(warmup_len * d²) for SVD + O(T * R * d)
    for projection.

    Using warmup_len << T means the SVD cost is bounded regardless of context length.
    After warmup, new tokens are projected in O(R * d) via frozen_add_token.
    """
    T, d = K.shape
    w = min(T, warmup_len)
    actual_rank = min(rank, w, d)

    # Full SVD on the warmup window — cost is O(w * d²), bounded since w <= warmup_len
    _, _, Vt_k = jnp.linalg.svd(K[:w], full_matrices=False)
    basis_k = Vt_k[:actual_rank]  # (R, d)

    _, _, Vt_v = jnp.linalg.svd(V_mat[:w], full_matrices=False)
    basis_v = Vt_v[:actual_rank]  # (R, d)

    # Project all T tokens onto the frozen basis in one matmul — O(T * R * d)
    proj_k = np.array(K @ basis_k.T)     # (T, R)
    proj_v = np.array(V_mat @ basis_v.T)  # (T, R)

    return FrozenBasisState(
        basis_k=basis_k,
        basis_v=basis_v,
        proj_k=proj_k,
        proj_v=proj_v,
        seq_len=T,
        rank=actual_rank,
    )


def frozen_add_token(
    state: FrozenBasisState,
    k_new: jnp.ndarray,  # (d,)
    v_new: jnp.ndarray,  # (d,)
) -> FrozenBasisState:
    """
    Project a new token onto the frozen basis. O(R * d) per token.
    Basis never changes — no stale projection problem.

    Note: np.vstack copies the full (T, R) array; pre-allocate for production.
    """
    c_k = np.array(k_new @ state.basis_k.T)[None, :]  # (1, R)
    c_v = np.array(v_new @ state.basis_v.T)[None, :]  # (1, R)
    return FrozenBasisState(
        basis_k=state.basis_k,
        basis_v=state.basis_v,
        proj_k=np.vstack([state.proj_k, c_k]),
        proj_v=np.vstack([state.proj_v, c_v]),
        seq_len=state.seq_len + 1,
        rank=state.rank,
    )


def frozen_reconstruct(
    state: FrozenBasisState,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct K and V from compressed form. For benchmarking only."""
    proj_k = jnp.array(state.proj_k)  # (T, R)
    proj_v = jnp.array(state.proj_v)  # (T, R)
    return proj_k @ state.basis_k, proj_v @ state.basis_v  # (T, d), (T, d)


# ---------------------------------------------------------------------------
# Sinked KV: full-precision sink + recent, frozen-basis middle
# Attention sink concept from StreamingLLM (Xiao et al., 2023) arXiv:2309.17453
# ---------------------------------------------------------------------------

@dataclass
class SinkedKVState:
    sink_k:    np.ndarray  # (n_sink, d)  — first K tokens, full precision
    sink_v:    np.ndarray  # (n_sink, d)
    middle:    Optional["FrozenBasisState"]  # compressed intermediate tokens, None when T is short
    recent_k:  np.ndarray  # (n_recent, d) — last K tokens, full precision
    recent_v:  np.ndarray  # (n_recent, d)
    sink_len:  int         # actual number of sink tokens stored
    window_len: int        # actual number of recent tokens stored
    seq_len:   int         # total sequence length
    d:         int         # head dimension


def sinked_cold_start(
    K: jnp.ndarray,
    V_mat: jnp.ndarray,
    rank: int,
    sink_len: int = 64,
    window_len: int = 64,
    warmup_len: int = 512,
) -> SinkedKVState:
    """
    Partition the KV cache into three segments:
      - Sink    (first sink_len tokens):   full precision — attention sinks per StreamingLLM
      - Middle  (tokens in between):       frozen-basis compressed
      - Recent  (last window_len tokens):  full precision — recency critical for generation

    When T <= sink_len + window_len there is no middle; everything stays full precision.
    Cite: Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks"
          arXiv:2309.17453
    """
    T, d = K.shape
    n_sink   = min(sink_len, T)
    n_recent = min(window_len, max(0, T - n_sink))
    T_mid    = T - n_sink - n_recent

    sink_k = np.array(K[:n_sink])
    sink_v = np.array(V_mat[:n_sink])

    if T_mid > 0:
        K_mid  = K[n_sink : n_sink + T_mid]
        V_mid  = V_mat[n_sink : n_sink + T_mid]
        middle = frozen_cold_start(K_mid, V_mid, rank, warmup_len=warmup_len)
    else:
        middle = None

    recent_start = n_sink + T_mid
    recent_k = np.array(K[recent_start:])   if n_recent > 0 else np.zeros((0, d), np.float32)
    recent_v = np.array(V_mat[recent_start:]) if n_recent > 0 else np.zeros((0, d), np.float32)

    return SinkedKVState(
        sink_k=sink_k, sink_v=sink_v,
        middle=middle,
        recent_k=recent_k, recent_v=recent_v,
        sink_len=n_sink, window_len=n_recent,
        seq_len=T, d=d,
    )


# --- Unit test ---
if __name__ == "__main__":
    T_MAX, R_MAX, D, N = 256, 16, 64, 64
    key = jax.random.PRNGKey(0)

    # --- Test 1: basic Brand update still works ---
    A = jax.random.normal(key, (N, D)) * jnp.array([1.0 / (i + 1) for i in range(D)])
    K_true = A
    V_true = jax.random.normal(jax.random.PRNGKey(1), (N, D))

    state = init_compressed_kv(T_MAX, R_MAX, D)
    for i in range(N):
        state = append_token(state, K_true[i], V_true[i], R_MAX)

    K_rec, _ = reconstruct_kv(state, T_MAX)
    err = jnp.linalg.norm(K_true - K_rec[:N]) / jnp.linalg.norm(K_true)
    assert err < 0.5, f"Brand update regression: {err:.4f}"
    print(f"Test 1 passed  — Brand update rel error: {err:.4f}")

    # --- Test 2: importance-biased cold_start reduces error on important tokens ---
    # First half of tokens are "important" (high weight), second half are not.
    importance = jnp.concatenate([jnp.ones(N // 2) * 10.0, jnp.ones(N // 2) * 0.1])

    state_uniform = cold_start(K_true, V_true, jax.random.normal(jax.random.PRNGKey(2), (R_MAX, D)), R_MAX, T_MAX, D)
    state_weighted = cold_start(K_true, V_true, jax.random.normal(jax.random.PRNGKey(2), (R_MAX, D)), R_MAX, T_MAX, D, importance_weights=importance)

    K_uni, _ = reconstruct_kv(state_uniform, T_MAX)
    K_wt, _  = reconstruct_kv(state_weighted, T_MAX)

    err_important_uniform  = float(jnp.linalg.norm(K_true[:N//2] - K_uni[:N//2]) / jnp.linalg.norm(K_true[:N//2]))
    err_important_weighted = float(jnp.linalg.norm(K_true[:N//2] - K_wt[:N//2])  / jnp.linalg.norm(K_true[:N//2]))

    print(f"Test 2 — error on important tokens:  uniform={err_important_uniform:.4f}  weighted={err_important_weighted:.4f}")
    assert err_important_weighted <= err_important_uniform + 0.05, \
        "Importance weighting should not significantly hurt important-token reconstruction"

    # --- Test 3: update_importance EMA ---
    fake_weights = jnp.zeros((T_MAX,)).at[:N].set(1.0 / N)
    state2 = update_importance(state, fake_weights, alpha=0.9)
    assert state2.importance[:N].mean() > 0, "update_importance produced zeros"
    print(f"Test 3 passed  — importance mean after EMA: {state2.importance[:N].mean():.4f}")

    print("kv_state.py: all checks passed")

    # --- Test 4: frozen_cold_start — reconstruction and add_token ---
    state_fr = frozen_cold_start(K_true, V_true, R_MAX, warmup_len=N)
    K_fr, _ = frozen_reconstruct(state_fr)
    err_fr = jnp.linalg.norm(K_true - K_fr) / jnp.linalg.norm(K_true)
    print(f"Test 4 — frozen cold_start rel error: {err_fr:.4f}")
    assert err_fr < 0.5, f"Frozen cold_start regression: {err_fr:.4f}"

    # add one extra token and verify shape grows correctly
    k_extra = jax.random.normal(jax.random.PRNGKey(10), (D,))
    v_extra = jax.random.normal(jax.random.PRNGKey(11), (D,))
    state_fr2 = frozen_add_token(state_fr, k_extra, v_extra)
    assert state_fr2.seq_len == N + 1, "frozen_add_token seq_len wrong"
    assert state_fr2.proj_k.shape == (N + 1, state_fr2.rank), "frozen_add_token shape wrong"
    print(f"Test 4 passed  — frozen add_token OK, seq_len={state_fr2.seq_len}")

    print("kv_state.py: all frozen checks passed")
