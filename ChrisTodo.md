# ChrisTodo

## Bug Fixes

### ~~loop.py — broken `inner_loop_update`~~ ✅
- ~~`np.ndarry` is a typo (`np.ndarray`), and `numpy` is never imported in the file~~
- ~~`@staticmethod` and `@jax.jit` are stacked in the wrong order — `@jax.jit` needs to wrap the function before `@staticmethod` takes it, otherwise JAX never sees the raw function~~
- ~~`d` and `w` are used inside `reshape()` but JAX traces them as dynamic values, which breaks shape inference — needs `static_argnums=(2, 3)` on the `jit` call~~

---

## Incomplete Implementations

### ~~`sinked_add_token()` missing from kv_state.py~~ ✅
~~Every other state type has an incremental append function (`append_token`, `append_token_lazy`, `frozen_add_token`), but `SinkedKVState` only has the batch `sinked_cold_start`. The streaming/autoregressive path for the sinked variant doesn't exist. Implement `sinked_add_token(state, k_new, v_new)` that routes the new token into the recent window and handles eviction of old recent tokens into the frozen middle.~~

### ~~`JaxTensorSketchStore.query()` missing from loop.py~~ ✅
~~The README describes the count-min sketch as a maintained layer, and the class has `init_state()` and `inner_loop_update()` but no way to query it. The standard count-min query is a pointwise min across hash rows — the class is half-built without it.~~

---

## Missing Plots

### ~~Metric 10 — `benchmark_sinked` has no plot~~ ✅
~~Every other benchmark in `benchmark.py` saves a PDF. `benchmark_sinked` (Metric 10) is called in `__main__` with only `print_sinked_table` — no `plot_sinked()` function exists. Needs a plot comparing frozen-only vs sinked reconstruction error and attention quality across sequence lengths.~~

---

## Research Task

### ~~Wire the count-min sketch into the KV state (core paper contribution)~~ ✅
~~The README and Plan.md both call out connecting the sketch to the KV cache as the central theoretical bridge — "Phi rows as the random sketch Omega for SVD connects the count-min sketch to optimal low-rank approximation." `sketch_to_range_basis()` in `loop.py` hints at this but it's never called from `cold_start()` or any attention path. This is the piece that ties `JaxTensorSketchStore` to `kv_state.py`.~~
