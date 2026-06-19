"""
OjaKV baseline: online low-rank KV cache compression via Oja's rule.

After every token, the right singular basis V is updated via one step of
the generalized Oja rule (gradient ascent on explained variance) followed
by QR re-orthonormalization.  Key/value vectors are stored as their
projections onto the *current* basis at the time of insertion; the basis
then rotates as new tokens arrive, so old projections become stale — the
primary source of approximation error vs BrandOnly's exact Brand update.

Interface mirrors kv_state.py so benchmark.py can call both identically.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class OjaKVState(NamedTuple):
    V_k:     jnp.ndarray  # (R_max, d) orthonormal right basis for keys
    V_v:     jnp.ndarray  # (R_max, d) orthonormal right basis for values
    K_proj:  jnp.ndarray  # (T_max, R_max) compressed keys (V_k @ k at insertion time)
    V_proj:  jnp.ndarray  # (T_max, R_max) compressed values
    seq_len: jnp.ndarray  # scalar int32


def init_oja_kv(T_max: int, R_max: int, d: int, seed: int = 0) -> OjaKVState:
    key = jax.random.PRNGKey(seed)
    # Random orthonormal rows as starting basis (same He-like scale as Phi_init)
    V_raw = jax.random.normal(key, (d, R_max)) * jnp.sqrt(2.0 / d)
    Q, _ = jnp.linalg.qr(V_raw)          # (d, R_max)
    V_init = Q[:, :R_max].T              # (R_max, d)
    return OjaKVState(
        V_k=V_init,
        V_v=V_init,
        K_proj=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        V_proj=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        seq_len=jnp.array(0, dtype=jnp.int32),
    )


def _oja_step(
    V: jnp.ndarray,        # (R_max, d) current orthonormal basis
    buf: jnp.ndarray,      # (T_max, R_max) projection buffer
    x: jnp.ndarray,        # (d,) new token vector
    seq_len: jnp.ndarray,  # scalar int32, position to write
    t: int,                # step count (for 1/(t+1) step size)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One Oja gradient step + QR orthonormalization, then store the compressed
    projection of x at position seq_len in buf.

    Step size eta = 1/(t+1) gives the diminishing schedule with convergence
    guarantees matching Oja (1982).  The h stored in buf is computed BEFORE
    the basis update so it reflects the basis that was active at insertion —
    matching how OjaKV stores projections during streaming decoding.
    """
    h = V @ x                               # (R_max,) — project onto current basis
    buf = jax.lax.dynamic_update_slice(buf, h[None, :], (seq_len, 0))

    eta = 1.0 / (t + 1)
    V_new = V + eta * jnp.outer(h, x)      # (R_max, d) gradient ascent step
    # Re-orthonormalize rows via economy QR on the transpose
    Q, R = jnp.linalg.qr(V_new.T)          # Q: (d, R_max), R: (R_max, R_max)
    signs = jnp.sign(jnp.diag(R))          # canonical sign (positive diagonal)
    V_orth = (Q * signs).T                 # (R_max, d)

    return V_orth, buf


def append_token_oja(
    state: OjaKVState,
    k_new: jnp.ndarray,   # (d,)
    v_new: jnp.ndarray,   # (d,)
    R_max: int,
) -> OjaKVState:
    t = int(state.seq_len)
    V_k, K_proj = _oja_step(state.V_k, state.K_proj, k_new, state.seq_len, t)
    V_v, V_proj = _oja_step(state.V_v, state.V_proj, v_new, state.seq_len, t)
    return OjaKVState(
        V_k=V_k, V_v=V_v,
        K_proj=K_proj, V_proj=V_proj,
        seq_len=state.seq_len + 1,
    )


def reconstruct_kv_oja(
    state: OjaKVState, T_max: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reconstruct K and V using current basis.  Because the basis rotated
    after each insertion, old projections are computed in a different basis,
    so K_proj[i] @ V_k != k_i in general — this mismatch is the error OjaKV
    accumulates over long sequences.
    """
    mask = (jnp.arange(T_max) < state.seq_len)[:, None]   # (T_max, 1)
    K = state.K_proj @ state.V_k                           # (T_max, d)
    V = state.V_proj @ state.V_v                           # (T_max, d)
    return jnp.where(mask, K, 0.0), jnp.where(mask, V, 0.0)


# --- Unit test ---
if __name__ == "__main__":
    import numpy as np

    T_MAX, R_MAX, D = 256, 16, 64
    key = jax.random.PRNGKey(0)

    A = jax.random.normal(key, (64, D)) * jnp.array([1.0 / (i + 1) for i in range(D)])
    K_true = A[:64]
    V_true = jax.random.normal(jax.random.PRNGKey(1), (64, D))

    state = init_oja_kv(T_MAX, R_MAX, D)
    for i in range(64):
        state = append_token_oja(state, K_true[i], V_true[i], R_MAX)

    K_rec, V_rec = reconstruct_kv_oja(state, T_MAX)
    K_rec = K_rec[:64]

    err_k = float(jnp.linalg.norm(K_true - K_rec) / jnp.linalg.norm(K_true))
    print(f"seq_len={state.seq_len}")
    print(f"K reconstruction rel error: {err_k:.4f}")
    print("oja_kv.py: unit test complete")
