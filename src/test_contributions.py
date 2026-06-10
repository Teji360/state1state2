"""
Smoke tests for Chris's contributions:
  1. loop.py bug fixes  (inner_loop_update + query)
  2. sinked_add_token   (kv_state.py)
  3. sketch_cold_start  (loop.py)
"""

import jax
import jax.numpy as jnp
import numpy as np

from kv_state import (
    sinked_cold_start,
    sinked_add_token,
    frozen_reconstruct,
)
from loop import JaxTensorSketchStore, sketch_cold_start
from attention import sinked_attention


# ---------------------------------------------------------------------------
# 1. loop.py bug fixes — inner_loop_update and query
# ---------------------------------------------------------------------------

def test_inner_loop_update():
    store = JaxTensorSketchStore(depth=4, width=8, embedding_dim=16)
    state = store.init_state()

    token = jax.random.normal(jax.random.PRNGKey(0), (16,))
    state2 = JaxTensorSketchStore.inner_loop_update(state, token, store.d, store.w)

    assert state2.S.shape == (4, 8), f"bad shape: {state2.S.shape}"
    assert not jnp.allclose(state2.S, state.S), "S should change after update"
    print("PASS  inner_loop_update — shape correct, state mutated")


def test_query():
    store = JaxTensorSketchStore(depth=4, width=8, embedding_dim=16)
    state = store.init_state()

    # Insert two tokens then query one of them back
    t1 = jax.random.normal(jax.random.PRNGKey(1), (16,))
    t2 = jax.random.normal(jax.random.PRNGKey(2), (16,))
    state = JaxTensorSketchStore.inner_loop_update(state, t1, store.d, store.w)
    state = JaxTensorSketchStore.inner_loop_update(state, t2, store.d, store.w)

    score_t1 = JaxTensorSketchStore.query(state, t1, store.d, store.w)
    score_zero = JaxTensorSketchStore.query(state, jnp.zeros(16), store.d, store.w)

    assert score_t1.shape == (), f"query should return scalar, got {score_t1.shape}"
    assert float(score_zero) == 0.0, "zero vector should score 0"
    print(f"PASS  query — scalar output, t1 score={float(score_t1):.4f}, zero score={float(score_zero):.4f}")


# ---------------------------------------------------------------------------
# 2. sinked_add_token
# ---------------------------------------------------------------------------

def test_sinked_add_token_streaming():
    """After cold_start with a full sequence, stream 10 tokens and verify
    that seq_len grows, window stays fixed, and middle grows by 10."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 50
    key = jax.random.PRNGKey(0)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW, warmup_len=16)
    assert state.middle is not None
    mid_start = state.middle.seq_len

    for i in range(10):
        k_new = jax.random.normal(jax.random.PRNGKey(100 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(200 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    assert state.seq_len == T + 10,         f"seq_len={state.seq_len}, expected {T+10}"
    assert state.sink_len == SINK,           f"sink changed: {state.sink_len}"
    assert state.window_len == WINDOW,       f"window changed: {state.window_len}"
    assert len(state.recent_k) == WINDOW,    f"recent_k len={len(state.recent_k)}"
    assert state.middle.seq_len == mid_start + 10, \
        f"middle grew by {state.middle.seq_len - mid_start}, expected 10"
    print(f"PASS  sinked_add_token streaming — seq_len={state.seq_len}, middle={state.middle.seq_len}")


def test_sinked_add_token_first_eviction():
    """Start with a short sequence (no middle), fill the window, trigger
    the first eviction, and confirm a middle is created with seq_len=1."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 12  # T < SINK + WINDOW so middle starts as None
    key = jax.random.PRNGKey(10)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(11), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW)
    assert state.middle is None, "middle should be None for short sequence"
    initial_window = state.window_len

    # Add enough tokens to fill the window and cause one eviction
    n_to_fill = WINDOW - initial_window + 1
    for i in range(n_to_fill):
        k_new = jax.random.normal(jax.random.PRNGKey(300 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(400 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    assert state.window_len == WINDOW,   f"window={state.window_len}"
    assert state.middle is not None,     "middle should exist after first eviction"
    assert state.middle.seq_len == 1,    f"middle.seq_len={state.middle.seq_len}, expected 1"
    print(f"PASS  sinked_add_token first eviction — middle created, seq_len={state.middle.seq_len}")


def test_sinked_add_token_attention():
    """Verify sinked_attention runs without error on a streamed state."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 50
    key = jax.random.PRNGKey(20)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(21), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW, warmup_len=16)
    for i in range(5):
        k_new = jax.random.normal(jax.random.PRNGKey(500 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(600 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    Q = jax.random.normal(jax.random.PRNGKey(99), (4, D)) * 0.1
    out = sinked_attention(Q, state)
    assert out.shape == (4, D), f"bad output shape: {out.shape}"
    assert not jnp.any(jnp.isnan(out)), "NaN in attention output"
    print(f"PASS  sinked_add_token attention — output shape {out.shape}, no NaNs")


# ---------------------------------------------------------------------------
# 3. sketch_cold_start
# ---------------------------------------------------------------------------

def test_sketch_cold_start_shape():
    """sketch_cold_start should return a CompressedKVState with the right shapes."""
    from kv_state import CompressedKVState, reconstruct_kv

    D, R_MAX, T_MAX = 32, 4, 64
    T = 20
    DEPTH, WIDTH = 2, R_MAX  # depth*width >= R_MAX

    key = jax.random.PRNGKey(0)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D)
    sketch_state = store.init_state()

    # Feed K rows into the sketch so it has seen the same tokens
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)
    assert isinstance(state, CompressedKVState), "wrong return type"
    assert state.U_k.shape == (T_MAX, R_MAX), f"U_k shape {state.U_k.shape}"
    assert state.V_k.shape == (R_MAX, D),     f"V_k shape {state.V_k.shape}"
    assert int(state.seq_len) == T,            f"seq_len={state.seq_len}"
    print(f"PASS  sketch_cold_start shapes — U_k={state.U_k.shape}, rank={int(state.active_rank)}")


def test_sketch_cold_start_reconstruction():
    """Reconstruction error should be reasonable (< 0.8) for low-rank data."""
    from kv_state import reconstruct_kv

    D, R_MAX, T_MAX = 32, 8, 64
    T = 30
    DEPTH, WIDTH = 2, R_MAX

    # Low-rank K so compression should work well
    key = jax.random.PRNGKey(5)
    base = jax.random.normal(key, (4, D))
    K = jax.random.normal(jax.random.PRNGKey(6), (T, 4)) @ base
    V = jax.random.normal(jax.random.PRNGKey(7), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D)
    sketch_state = store.init_state()
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)
    K_rec, _ = reconstruct_kv(state, T_MAX)
    err = float(jnp.linalg.norm(K - K_rec[:T]) / (jnp.linalg.norm(K) + 1e-9))
    assert err < 0.8, f"reconstruction error too large: {err:.4f}"
    print(f"PASS  sketch_cold_start reconstruction — rel error={err:.4f}")


def test_sketch_cold_start_importance_effect():
    """With full-rank data (lossy compression), sketch_cold_start should
    produce valid output and not diverge vs plain cold_start on the same phi."""
    from kv_state import cold_start, reconstruct_kv

    D, R_MAX, T_MAX = 32, 4, 64  # R_MAX < D so compression is genuinely lossy
    T = 30
    DEPTH, WIDTH = 2, R_MAX

    key = jax.random.PRNGKey(9)
    K = jax.random.normal(key, (T, D))  # full-rank: compression must approximate
    V = jax.random.normal(jax.random.PRNGKey(11), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D, seed=12)
    sketch_state = store.init_state()
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state_plain  = cold_start(K, V, store.Phi_init, R_MAX, T_MAX, D)
    state_sketch = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)

    K_plain,  _ = reconstruct_kv(state_plain,  T_MAX)
    K_sketch, _ = reconstruct_kv(state_sketch, T_MAX)
    err_plain  = float(jnp.linalg.norm(K - K_plain[:T])  / (jnp.linalg.norm(K) + 1e-9))
    err_sketch = float(jnp.linalg.norm(K - K_sketch[:T]) / (jnp.linalg.norm(K) + 1e-9))

    # Both are lossy — sketch-weighted should be in the same ballpark as plain.
    assert err_sketch < err_plain + 0.15, \
        f"sketch err={err_sketch:.4f} is more than 0.15 above plain err={err_plain:.4f}"
    print(f"PASS  sketch importance effect — plain={err_plain:.4f}, sketch={err_sketch:.4f}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_inner_loop_update,
        test_query,
        test_sinked_add_token_streaming,
        test_sinked_add_token_first_eviction,
        test_sinked_add_token_attention,
        test_sketch_cold_start_shape,
        test_sketch_cold_start_reconstruction,
        test_sketch_cold_start_importance_effect,
    ]

    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"FAIL  {t.__name__} — {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)
