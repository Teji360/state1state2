"""
Three benchmark metrics for the NeurIPS workshop paper:

  1. Reconstruction error  — ||K - K_r||_F / ||K||_F vs rank & seq_len
  2. Memory compression ratio — theoretical bytes saved vs rank
  3. Perplexity on short texts vs rank — measured on GPT-2 small

Run: python benchmark.py
Outputs: benchmark_reconstruction.pdf, benchmark_memory.pdf, benchmark_perplexity.pdf
"""

import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kv_state import (
    init_compressed_kv, append_token, reconstruct_kv,
    init_lazy_kv, append_token_lazy, reconstruct_kv_lazy,
    cold_start,
    TKVState,
    tkv_cold_start, tkv_add_token, tkv_reconstruct,
    brand_to_tkv, hybrid_append_token, init_tkv_streaming,
    SinkedKVState, sinked_cold_start,
)
from attention import tkv_compressed_attention, sinked_attention
from oja_kv import init_oja_kv, append_token_oja, reconstruct_kv_oja
from model_hooks import load_gpt2


def extract_real_kv(max_seq_len: int = 512, layer_idx: int = 6, head_idx: int = 0):
    """
    Capture real K/V activations from GPT-2 small on a WikiText-2 passage.

    Runs a single forward pass with a lightweight capture hook registered in
    ALL_ATTENTION_FUNCTIONS. The hook saves key/value for `layer_idx` and
    passes through exact SDPA attention for every layer so downstream activations
    are correct.

    Returns (K, V) as JAX arrays of shape (max_seq_len, head_dim=64).
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS
    from datasets import load_dataset

    model, tok = load_gpt2()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text, return_tensors="pt")[:, :max_seq_len]

    captured = {}
    call_count = [0]

    def _capture_fn(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
        layer = call_count[0]
        call_count[0] += 1
        if layer == layer_idx:
            captured["k"] = key[0, head_idx].detach().float().cpu()
            captured["v"] = value[0, head_idx].detach().float().cpu()
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        out = F.scaled_dot_product_attention(query, key, value, scale=scale, is_causal=True)
        return out.transpose(1, 2), None  # (batch, seq_len, heads, head_dim)

    IMPL_KEY = "_tkv_capture"
    ALL_ATTENTION_FUNCTIONS.register(IMPL_KEY, _capture_fn)
    orig_impl = model.config._attn_implementation
    model.config._attn_implementation = IMPL_KEY

    with torch.no_grad():
        model(ids, use_cache=False)

    model.config._attn_implementation = orig_impl

    K = jnp.array(captured["k"].numpy())  # (max_seq_len, head_dim)
    V = jnp.array(captured["v"].numpy())
    return K, V


# ---------------------------------------------------------------------------
# Metric 1: Reconstruction error
# ---------------------------------------------------------------------------

def benchmark_reconstruction(
    seq_lens=(32, 64, 128, 256, 512),
    ranks=(2, 4, 8, 16, 32),
    d=64,
    T_max=1024,
):
    """
    For each (seq_len, rank): build compressed state by appending tokens one-by-one
    (Brand update), then measure ||K - K_r||_F / ||K||_F.

    Uses real K/V activations from GPT-2 small (layer 6, head 0) on WikiText-2.
    """
    print("Extracting real K/V from GPT-2...")
    K_real, V_real = extract_real_kv(max_seq_len=max(seq_lens))
    results = {}  # (seq_len, rank) -> rel_error

    for seq_len in seq_lens:
        K_true = K_real[:seq_len]
        V_true = V_real[:seq_len]

        for rank in ranks:
            state = init_compressed_kv(T_max, rank, d)
            for i in range(seq_len):
                state = append_token(state, K_true[i], V_true[i], rank)

            K_rec, _ = reconstruct_kv(state, T_max)
            K_rec = K_rec[:seq_len]
            rel_err = float(
                jnp.linalg.norm(K_true - K_rec) / (jnp.linalg.norm(K_true) + 1e-9)
            )
            results[(seq_len, rank)] = rel_err

    return results


def plot_reconstruction(results, seq_lens, ranks, path="benchmark_reconstruction.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: error vs rank for several seq_lens
    ax = axes[0]
    for seq_len in seq_lens[::2]:
        errs = [results[(seq_len, r)] for r in ranks]
        ax.plot(ranks, errs, marker="o", label=f"t={seq_len}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Reconstruction Error vs Rank")
    ax.legend()
    ax.set_yscale("log")

    # Right: error vs seq_len for several ranks
    ax = axes[1]
    for rank in ranks:
        errs = [results[(t, rank)] for t in seq_lens]
        ax.plot(seq_lens, errs, marker="s", label=f"r={rank}")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Reconstruction Error vs Sequence Length")
    ax.legend()
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Metric 2: Memory compression ratio
# ---------------------------------------------------------------------------

def memory_ratio(seq_len: int, d: int, rank: int) -> float:
    """
    Theoretical bytes (float32) for compressed vs naive KV cache (K and V).
    Compressed: 2 * (seq_len*rank + rank + rank*d)  [U, sigma, V for K and V]
    Naive:      2 * seq_len * d
    T_max is the pre-allocated buffer size; we report *occupancy* (seq_len rows),
    since the paper claims per-sequence memory not per-buffer allocation.
    """
    compressed = 2 * (seq_len * rank + rank + rank * d)
    naive = 2 * seq_len * d
    return compressed / naive


def benchmark_memory(
    seq_lens=(64, 128, 256, 512, 1024),
    ranks=(2, 4, 8, 16, 32),
    d=64,
):
    table = {}
    for seq_len in seq_lens:
        for rank in ranks:
            table[(seq_len, rank)] = memory_ratio(seq_len, d, rank)
    return table


def plot_memory(table, seq_lens, ranks, path="benchmark_memory.pdf"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rank in ranks:
        ratios = [table[(t, rank)] for t in seq_lens]
        ax.plot(seq_lens, ratios, marker="o", label=f"r={rank}")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="baseline (exact)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Memory Ratio (compressed / exact)")
    ax.set_title("KV Cache Memory Ratio vs Sequence Length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # Print summary table
    header = "seq_len |" + "".join(f"  r={r:2d}" for r in ranks)
    print(header)
    print("-" * len(header))
    for t in seq_lens:
        row = f"{t:7d} |" + "".join(f"  {table[(t,r)]:.3f}" for r in ranks)
        print(row)


# ---------------------------------------------------------------------------
# Metric 3: Perplexity on GPT-2
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The transformer architecture has fundamentally changed how we approach natural language processing tasks.",
    "Singular value decomposition provides the optimal low-rank approximation under the Frobenius norm.",
    "Memory efficiency in large language models is critical for deployment at scale.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "The key-value cache stores intermediate computations to speed up autoregressive generation.",
]


def benchmark_perplexity(
    ranks=(4, 8, 16, 32, 48),
    max_length=128,
    T_max=512,
    head_dim=64,
    device="cpu",
):
    from model_hooks import (
        load_gpt2, make_phi_for_gpt2, compute_perplexity,
        patch_gpt2_for_compression, unpatch_gpt2,
    )

    print("Loading GPT-2...")
    model, tokenizer = load_gpt2(device)

    ppl_exact = compute_perplexity(
        model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device
    )
    print(f"  Exact perplexity: {ppl_exact:.2f}")

    results = {"exact": ppl_exact}
    for rank in ranks:
        phi = make_phi_for_gpt2(rank, head_dim)
        orig = patch_gpt2_for_compression(model, phi, rank, T_max)
        ppl = compute_perplexity(
            model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device
        )
        unpatch_gpt2(model, orig)
        results[rank] = ppl
        print(f"  rank={rank:3d}: perplexity={ppl:.2f}  ratio={ppl/ppl_exact:.4f}")

    return results


def plot_perplexity(results, path="benchmark_perplexity.pdf"):
    ranks = [k for k in results if k != "exact"]
    ppl_exact = results["exact"]
    ppls = [results[r] for r in ranks]
    ratios = [p / ppl_exact for p in ppls]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ranks, ppls, marker="o", color="steelblue")
    ax.axhline(ppl_exact, color="k", linestyle="--", linewidth=0.8, label="exact")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Perplexity")
    ax.set_title("GPT-2 Perplexity vs KV Cache Rank")
    ax.legend()

    ax = axes[1]
    ax.plot(ranks, ratios, marker="s", color="tomato")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Perplexity Ratio (compressed / exact)")
    ax.set_title("Perplexity Degradation vs Rank")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def benchmark_wikitext_perplexity(
    ranks=(8, 16, 32, 48),
    context_len=512,
    stride=128,
    n_windows=30,
    T_max=1024,
    head_dim=64,
    device="cpu",
):
    """
    Sliding-window perplexity on WikiText-2 test set for the Halko cold-start
    compressed attention (single SVD per forward call, no Brand streaming).
    Replaces the held-out sentence evaluation with a standard reproducible benchmark.
    """
    import math
    import torch
    from datasets import load_dataset
    from model_hooks import (
        load_gpt2, make_phi_for_gpt2, patch_gpt2_for_compression, unpatch_gpt2,
    )

    print("Loading WikiText-2 test set...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    print("Loading GPT-2...")
    model, tokenizer = load_gpt2(device)

    @torch.no_grad()
    def _sliding_ppl(mdl):
        ids = tokenizer.encode(text)
        total_nll, total_toks = 0.0, 0
        for start in range(0, min(len(ids) - context_len, stride * n_windows), stride):
            chunk = torch.tensor([ids[start:start + context_len]], dtype=torch.long, device=device)
            logits = mdl(chunk, use_cache=False).logits[0]
            tgt_start = context_len - stride
            log_probs = torch.log_softmax(logits[tgt_start:-1].float(), dim=-1)
            targets = chunk[0, tgt_start + 1:]
            nll = -log_probs[range(len(targets)), targets].mean().item()
            total_nll += nll * len(targets)
            total_toks += len(targets)
        return math.exp(total_nll / total_toks) if total_toks > 0 else float("inf")

    ppl_exact = _sliding_ppl(model)
    print(f"  WikiText-2 exact perplexity: {ppl_exact:.2f}")
    results = {"exact": ppl_exact}

    for rank in ranks:
        phi = make_phi_for_gpt2(rank, head_dim)
        orig = patch_gpt2_for_compression(model, phi, rank, T_max)
        ppl = _sliding_ppl(model)
        unpatch_gpt2(model, orig)
        results[rank] = ppl
        print(f"  rank={rank:3d}: ppl={ppl:.2f}  ratio={ppl/ppl_exact:.4f}  mem={2*(context_len*rank+rank+rank*head_dim)/(2*context_len*head_dim):.3f}")

    return results


# ---------------------------------------------------------------------------
# Metric 4: TKV vs OjaKV head-to-head
# ---------------------------------------------------------------------------

def benchmark_brand_vs_oja(
    seq_lens=(32, 64, 128, 256, 512),
    ranks=(2, 4, 8, 16, 32),
    d=64,
    T_max=1024,
):
    """
    Same K matrix, same rank budget, same sequence lengths.
    Returns dicts keyed by (seq_len, rank) for each method.

    Uses real K/V activations from GPT-2 small (layer 6, head 0) on WikiText-2.
    """
    print("Extracting real K/V from GPT-2...")
    K_real, V_real = extract_real_kv(max_seq_len=max(seq_lens))
    brand_err, oja_err = {}, {}
    brand_time, oja_time = {}, {}

    for seq_len in seq_lens:
        K_true = K_real[:seq_len]
        V_true = V_real[:seq_len]

        for rank in ranks:
            # --- BrandOnly ---
            state_brand = init_compressed_kv(T_max, rank, d)
            t0 = time.perf_counter()
            for i in range(seq_len):
                state_brand = append_token(state_brand, K_true[i], V_true[i], rank)
            jax.block_until_ready(state_brand.U_k)
            brand_time[(seq_len, rank)] = (time.perf_counter() - t0) / seq_len

            K_rec, _ = reconstruct_kv(state_brand, T_max)
            brand_err[(seq_len, rank)] = float(
                jnp.linalg.norm(K_true - K_rec[:seq_len])
                / (jnp.linalg.norm(K_true) + 1e-9)
            )

            # --- OjaKV ---
            state_o = init_oja_kv(T_max, rank, d)
            t0 = time.perf_counter()
            for i in range(seq_len):
                state_o = append_token_oja(state_o, K_true[i], V_true[i], rank)
            jax.block_until_ready(state_o.V_k)
            oja_time[(seq_len, rank)] = (time.perf_counter() - t0) / seq_len

            K_rec_o, _ = reconstruct_kv_oja(state_o, T_max)
            oja_err[(seq_len, rank)] = float(
                jnp.linalg.norm(K_true - K_rec_o[:seq_len])
                / (jnp.linalg.norm(K_true) + 1e-9)
            )

    return brand_err, oja_err, brand_time, oja_time


def _run_one_head_comparison(K_head, V_head, seq_len, rank, T_max=1024, d=64):
    """Run BrandOnly vs OjaKV on one (K, V) pair; return (brand_rel_err, oja_rel_err)."""
    K_true = K_head[:seq_len]
    V_true = V_head[:seq_len]

    state_brand = init_compressed_kv(T_max, rank, d)
    for i in range(seq_len):
        state_brand = append_token(state_brand, K_true[i], V_true[i], rank)
    K_rec, _ = reconstruct_kv(state_brand, T_max)
    brand_e = float(jnp.linalg.norm(K_true - K_rec[:seq_len]) / (jnp.linalg.norm(K_true) + 1e-9))

    state_o = init_oja_kv(T_max, rank, d)
    for i in range(seq_len):
        state_o = append_token_oja(state_o, K_true[i], V_true[i], rank)
    K_rec_o, _ = reconstruct_kv_oja(state_o, T_max)
    oja_e = float(jnp.linalg.norm(K_true - K_rec_o[:seq_len]) / (jnp.linalg.norm(K_true) + 1e-9))

    return brand_e, oja_e


def benchmark_multi_head_comparison(
    layers=(0, 3, 6, 9, 11),
    heads=(0, 4, 8),
    seq_lens=(64, 128, 256, 512),
    ranks=(8, 16, 32),
    d=64,
    T_max=1024,
):
    """
    Sweep BrandOnly vs OjaKV over multiple GPT-2 small layers and heads.
    Reports mean ± std of reconstruction error and advantage across all configs.
    15 configurations = 5 layers × 3 heads.
    """
    import numpy as np

    configs = [(l, h) for l in layers for h in heads]
    max_len = max(seq_lens)

    all_brand = {(t, r): [] for t in seq_lens for r in ranks}
    all_oja = {(t, r): [] for t in seq_lens for r in ranks}

    for layer_idx, head_idx in configs:
        print(f"  Extracting layer={layer_idx}, head={head_idx} ...", flush=True)
        K, V = extract_real_kv(max_seq_len=max_len, layer_idx=layer_idx, head_idx=head_idx)
        for seq_len in seq_lens:
            for rank in ranks:
                be, oe = _run_one_head_comparison(K, V, seq_len, rank, T_max, d)
                all_brand[(seq_len, rank)].append(be)
                all_oja[(seq_len, rank)].append(oe)

    print("\nMulti-head BrandOnly vs OjaKV (mean ± std over {} configs)".format(len(configs)))
    hdr = f"{'T':>4} {'R':>3}  {'BrandOnly':^16}  {'OjaKV':^16}  {'Advantage':^14}"
    print(hdr)
    print("-" * len(hdr))
    for seq_len in seq_lens:
        for rank in ranks:
            brand_arr = np.array(all_brand[(seq_len, rank)])
            oja_arr = np.array(all_oja[(seq_len, rank)])
            adv_arr = oja_arr / (brand_arr + 1e-9)
            print(
                f"{seq_len:4d} {rank:3d}  "
                f"{brand_arr.mean():.3f}±{brand_arr.std():.3f}  "
                f"{oja_arr.mean():.3f}±{oja_arr.std():.3f}  "
                f"{adv_arr.mean():.2f}×±{adv_arr.std():.2f}×"
            )

    return all_brand, all_oja


def benchmark_streaming_error(rank=16, seq_len=512, d=64, T_max=1024):
    """
    Records reconstruction error after every token for both methods.
    Shows how error accumulates as the sequence grows.

    Uses real K/V activations from GPT-2 small (layer 6, head 0) on WikiText-2.
    """
    K_true, V_true = extract_real_kv(max_seq_len=seq_len)

    state_brand = init_compressed_kv(T_max, rank, d)
    state_o = init_oja_kv(T_max, rank, d)

    brand_curve, oja_curve = [], []
    for i in range(seq_len):
        state_brand = append_token(state_brand, K_true[i], V_true[i], rank)
        state_o = append_token_oja(state_o, K_true[i], V_true[i], rank)

        if (i + 1) % 8 == 0:
            t = i + 1
            K_t, _ = reconstruct_kv(state_brand, T_max)
            K_o, _ = reconstruct_kv_oja(state_o, T_max)
            ref = float(jnp.linalg.norm(K_true[:t])) + 1e-9
            brand_curve.append((t, float(jnp.linalg.norm(K_true[:t] - K_t[:t])) / ref))
            oja_curve.append((t, float(jnp.linalg.norm(K_true[:t] - K_o[:t])) / ref))

    return brand_curve, oja_curve


def plot_comparison(
    brand_err, oja_err, brand_time, oja_time,
    brand_curve, oja_curve,
    seq_lens, ranks,
    path="benchmark_brand_vs_oja.pdf",
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: reconstruction error vs rank at longest seq_len
    ax = axes[0]
    t = seq_lens[-1]
    ax.plot(ranks, [brand_err[(t, r)] for r in ranks], marker="o", label="BrandOnly")
    ax.plot(ranks, [oja_err[(t, r)] for r in ranks], marker="s", linestyle="--", label="OjaKV (Oja)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title(f"Reconstruction Error vs Rank (t={t})")
    ax.set_yscale("log")
    ax.legend()

    # Panel 2: error accumulation over tokens (streaming)
    ax = axes[1]
    ts_t, errs_t = zip(*brand_curve)
    ts_o, errs_o = zip(*oja_curve)
    ax.plot(ts_t, errs_t, marker="o", markersize=3, label="BrandOnly")
    ax.plot(ts_o, errs_o, marker="s", markersize=3, linestyle="--", label="OjaKV (Oja)")
    ax.set_xlabel("Tokens Processed")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title(f"Streaming Error Accumulation (rank={ranks[-2]})")
    ax.set_yscale("log")
    ax.legend()

    # Panel 3: per-token update time vs rank
    ax = axes[2]
    t = seq_lens[-1]
    brand_times = [brand_time[(t, r)] * 1e3 for r in ranks]
    oja_times = [oja_time[(t, r)] * 1e3 for r in ranks]
    ax.plot(ranks, brand_times, marker="o", label="BrandOnly")
    ax.plot(ranks, oja_times, marker="s", linestyle="--", label="OjaKV (Oja)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("ms / token")
    ax.set_title(f"Per-Token Update Time (t={t})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def benchmark_flush_sweep(
    flush_every_vals=(1, 2, 4, 8),
    seq_lens=(64, 256, 512),
    ranks=(8, 16, 32),
    d=64,
    T_max=1024,
    seed=0,
):
    """
    For each flush_every value, measure reconstruction error and ms/tok vs
    exact BrandOnly (flush=1) and OjaKV.  Same data and rank budget throughout.
    """
    key = jax.random.PRNGKey(seed)
    results = {}   # (method_label, seq_len, rank) -> {err, ms_tok}

    for seq_len in seq_lens:
        k1, k2 = jax.random.split(key)
        decay = jnp.array([1.0 / (i + 1) ** 0.5 for i in range(d)])
        K_true = jax.random.normal(k1, (seq_len, d)) * decay
        V_true = jax.random.normal(k2, (seq_len, d))

        for rank in ranks:
            ref = float(jnp.linalg.norm(K_true)) + 1e-9

            # --- Exact BrandOnly baseline (flush_every=1 uses _brand_update directly) ---
            state_e = init_compressed_kv(T_max, rank, d)
            t0 = time.perf_counter()
            for i in range(seq_len):
                state_e = append_token(state_e, K_true[i], V_true[i], rank)
            jax.block_until_ready(state_e.U_k)
            ms_tok_e = (time.perf_counter() - t0) / seq_len * 1e3
            K_rec_e, _ = reconstruct_kv(state_e, T_max)
            err_e = float(jnp.linalg.norm(K_true - K_rec_e[:seq_len])) / ref
            results[("BrandOnly-exact", seq_len, rank)] = {"err": err_e, "ms_tok": ms_tok_e}

            # --- Lazy BrandOnly variants (flush_every > 1) ---
            for fe in [fe for fe in flush_every_vals if fe > 1]:
                label = f"BrandOnly-flush{fe}"
                state = init_lazy_kv(T_max, rank, d)
                t0 = time.perf_counter()
                for i in range(seq_len):
                    state = append_token_lazy(state, K_true[i], V_true[i], rank, flush_every=fe)
                jax.block_until_ready(state.U_k)
                ms_tok = (time.perf_counter() - t0) / seq_len * 1e3
                K_rec, _ = reconstruct_kv_lazy(state, T_max)
                err = float(jnp.linalg.norm(K_true - K_rec[:seq_len])) / ref
                results[(label, seq_len, rank)] = {"err": err, "ms_tok": ms_tok}

            # --- OjaKV ---
            state_o = init_oja_kv(T_max, rank, d)
            t0 = time.perf_counter()
            for i in range(seq_len):
                state_o = append_token_oja(state_o, K_true[i], V_true[i], rank)
            jax.block_until_ready(state_o.V_k)
            ms_tok = (time.perf_counter() - t0) / seq_len * 1e3
            K_rec_o, _ = reconstruct_kv_oja(state_o, T_max)
            err_o = float(jnp.linalg.norm(K_true - K_rec_o[:seq_len])) / ref
            results[("OjaKV", seq_len, rank)] = {"err": err_o, "ms_tok": ms_tok}

    return results


def plot_flush_sweep(results, flush_every_vals, seq_lens, ranks, path="benchmark_flush_sweep.pdf"):
    # One row per seq_len, two columns: error vs rank and ms/tok vs rank
    n = len(seq_lens)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    oja_color = "tomato"

    for row, seq_len in enumerate(seq_lens):
        ax_err, ax_spd = axes[row]

        for ci, fe in enumerate(flush_every_vals):
            label = f"BrandOnly-flush{fe}" if fe > 1 else "BrandOnly-exact"
            errs = [results[(label, seq_len, r)]["err"] for r in ranks]
            spds = [results[(label, seq_len, r)]["ms_tok"] for r in ranks]
            ls = "-" if fe == 1 else "--"
            ax_err.plot(ranks, errs, marker="o", color=colors[ci], linestyle=ls, label=label)
            ax_spd.plot(ranks, spds, marker="o", color=colors[ci], linestyle=ls, label=label)

        oja_errs = [results[("OjaKV", seq_len, r)]["err"] for r in ranks]
        oja_spds = [results[("OjaKV", seq_len, r)]["ms_tok"] for r in ranks]
        ax_err.plot(ranks, oja_errs, marker="s", color=oja_color, linestyle=":", label="OjaKV")
        ax_spd.plot(ranks, oja_spds, marker="s", color=oja_color, linestyle=":", label="OjaKV")

        ax_err.set_title(f"Reconstruction Error (t={seq_len})")
        ax_err.set_xlabel("Rank")
        ax_err.set_ylabel("Relative Frobenius Error")
        ax_err.set_yscale("log")
        ax_err.legend(fontsize=8)

        ax_spd.set_title(f"Per-Token Time (t={seq_len})")
        ax_spd.set_xlabel("Rank")
        ax_spd.set_ylabel("ms / token")
        ax_spd.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def print_flush_table(results, flush_every_vals, seq_len, ranks):
    labels = [f"BrandOnly-flush{fe}" if fe > 1 else "BrandOnly-exact" for fe in flush_every_vals] + ["OjaKV"]
    header = f"{'rank':>5}" + "".join(f"  {l:>14}" for l in labels)
    print(f"\nt={seq_len} — error / ms_tok")
    print(header)
    print("-" * len(header))
    for rank in ranks:
        row = f"{rank:>5}"
        for label in labels:
            r = results[(label, seq_len, rank)]
            row += f"  {r['err']:.3f}/{r['ms_tok']:5.2f}ms"
        print(row)


def print_comparison_table(brand_err, oja_err, brand_time, oja_time, seq_lens, ranks):
    print(f"\n{'seq':>5} {'rank':>4}  {'Brand err':>9}  {'Oja err':>9}  {'Brand ms/tok':>10}  {'Oja ms/tok':>10}  {'err ratio':>9}")
    print("-" * 70)
    for seq_len in seq_lens:
        for rank in ranks:
            be = brand_err[(seq_len, rank)]
            oe = oja_err[(seq_len, rank)]
            bt = brand_time[(seq_len, rank)] * 1e3
            ot = oja_time[(seq_len, rank)] * 1e3
            ratio = oe / (be + 1e-12)
            print(f"{seq_len:>5} {rank:>4}  {be:>9.4f}  {oe:>9.4f}  {bt:>10.3f}  {ot:>10.3f}  {ratio:>9.2f}x")


# ---------------------------------------------------------------------------
# Metric 6: Importance-Weighted Cold Start (Moment-KV integration)
# ---------------------------------------------------------------------------

def benchmark_importance(
    ranks=(4, 8, 16, 32),
    seq_len=128,
    d=64,
    T_max=512,
    n_important=32,
    seed=0,
):
    """
    Constructs a K matrix where the first n_important tokens share a low-rank
    subspace (rank-4) and the remaining tokens are random noise.  This mimics
    real attention patterns where high-attention tokens cluster semantically.

    Importance weights are oracle: high for important tokens, low for noise.
    We compare cold_start with and without importance weights at each rank.

    Metrics reported per rank:
      - err_important_uniform:  reconstruction error on important tokens, no weighting
      - err_important_weighted: reconstruction error on important tokens, with weighting
      - err_total_uniform:      total reconstruction error, no weighting
      - err_total_weighted:     total reconstruction error, with weighting
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Important tokens: low-rank structure (rank-4 subspace)
    basis = jax.random.normal(k1, (4, d)) * 0.5
    basis = basis / jnp.linalg.norm(basis, axis=1, keepdims=True)
    coefs_imp = jax.random.normal(k2, (n_important, 4))
    K_important = coefs_imp @ basis                          # (n_important, d)

    # Unimportant tokens: random, independent of the important subspace
    K_noise = jax.random.normal(k3, (seq_len - n_important, d)) * 0.1

    K_true = jnp.concatenate([K_important, K_noise], axis=0)   # (seq_len, d)
    V_true = jax.random.normal(k4, (seq_len, d)) * 0.1

    # Oracle importance: high for important tokens, low for noise
    importance = jnp.concatenate([
        jnp.ones(n_important) * 10.0,
        jnp.ones(seq_len - n_important) * 0.1,
    ])

    phi = jax.random.normal(jax.random.PRNGKey(99), (max(ranks), d)) * jnp.sqrt(2.0 / d)

    results = {}
    for rank in ranks:
        phi_r = phi[:rank]

        s_uni = cold_start(K_true, V_true, phi_r, rank, T_max, d)
        s_wt  = cold_start(K_true, V_true, phi_r, rank, T_max, d,
                           importance_weights=importance)

        K_uni, _ = reconstruct_kv(s_uni, T_max)
        K_wt,  _ = reconstruct_kv(s_wt,  T_max)

        def rel_err(K_rec, rows):
            num = jnp.linalg.norm(K_true[:rows] - K_rec[:rows])
            den = jnp.linalg.norm(K_true[:rows]) + 1e-9
            return float(num / den)

        results[rank] = {
            "err_imp_uni": rel_err(K_uni, n_important),
            "err_imp_wt":  rel_err(K_wt,  n_important),
            "err_tot_uni": rel_err(K_uni, seq_len),
            "err_tot_wt":  rel_err(K_wt,  seq_len),
        }

    return results


def plot_importance(results, ranks, path="benchmark_importance.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ranks, [results[r]["err_imp_uni"] for r in ranks],
            marker="o", label="uniform")
    ax.plot(ranks, [results[r]["err_imp_wt"]  for r in ranks],
            marker="s", linestyle="--", label="importance-weighted")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Important-Token Reconstruction Error vs Rank")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(ranks, [results[r]["err_tot_uni"] for r in ranks],
            marker="o", label="uniform")
    ax.plot(ranks, [results[r]["err_tot_wt"]  for r in ranks],
            marker="s", linestyle="--", label="importance-weighted")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Total Reconstruction Error vs Rank")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# TKV (warmup-then-freeze) long-context benchmarks (128K+ target)
# ---------------------------------------------------------------------------

# GPT-2 small geometry used for memory calculations
_GPT2_LAYERS = 12
_GPT2_HEADS  = 12

PYRAMID_RANKS_GPT2 = [32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4]


def _tkv_bytes(seq_len: int, rank: int, d: int) -> int:
    """Bytes for one head: proj_k (T×R) + proj_v (T×R) + basis_k (R×d) + basis_v (R×d), float32."""
    return 4 * (2 * seq_len * rank + 2 * rank * d)


def benchmark_hybrid_streaming(
    seq_lens=(128, 512, 1_000, 4_000, 16_000),
    rank=16,
    warmup_len=512,
    d=64,
    seed=0,
):
    """
    Simulates token-by-token streaming for three methods:
      - BrandOnly: exact SVD throughout, O(R³ + T·R²) per token
      - TKV only:  SVD on first warmup_len tokens, O(R·d) after
      - Hybrid:    Brand's until warmup_len, then TKV freeze (brand_to_tkv)

    Reports reconstruction error and total update time at each seq_len.
    """
    key = jax.random.PRNGKey(seed)
    results = {}

    for seq_len in seq_lens:
        k1, k2 = jax.random.split(key)
        decay = jnp.array([1.0 / (i + 1) ** 0.5 for i in range(d)])
        K_true = jax.random.normal(k1, (seq_len, d)) * decay
        V_true = jax.random.normal(k2, (seq_len, d))
        ref = float(jnp.linalg.norm(K_true)) + 1e-9

        T_max = seq_len + 1

        # Brand's only (streaming)
        t0 = time.perf_counter()
        state_b = init_compressed_kv(T_max, rank, d)
        for i in range(seq_len):
            state_b = append_token(state_b, K_true[i], V_true[i], rank)
        jax.block_until_ready(state_b.U_k)
        brand_ms = (time.perf_counter() - t0) * 1e3
        K_b, _ = reconstruct_kv(state_b, T_max)
        brand_err = float(jnp.linalg.norm(K_true - K_b[:seq_len])) / ref

        # TKV only (batch SVD on warmup, then stream)
        t0 = time.perf_counter()
        state_f = tkv_cold_start(K_true[:min(seq_len, warmup_len)],
                                 V_true[:min(seq_len, warmup_len)], rank)
        for i in range(min(seq_len, warmup_len), seq_len):
            state_f = tkv_add_token(state_f, K_true[i], V_true[i])
        tkv_ms = (time.perf_counter() - t0) * 1e3
        K_f, _ = tkv_reconstruct(state_f)
        tkv_err = float(jnp.linalg.norm(K_true - K_f)) / ref

        # Hybrid (Brand's until warmup_len, then auto-converts to TKV freeze)
        t0 = time.perf_counter()
        state_h = init_compressed_kv(T_max, rank, d)
        for i in range(seq_len):
            state_h = hybrid_append_token(state_h, K_true[i], V_true[i], rank,
                                          warmup_len=warmup_len)
        hybrid_ms = (time.perf_counter() - t0) * 1e3
        K_h, _ = tkv_reconstruct(state_h) if isinstance(state_h, TKVState) \
                  else reconstruct_kv(state_h, T_max)
        hybrid_err = float(jnp.linalg.norm(K_true - K_h[:seq_len])) / ref

        results[seq_len] = {
            "brand_err": brand_err,  "brand_ms": brand_ms,
            "tkv_err": tkv_err, "tkv_ms": tkv_ms,
            "hybrid_err": hybrid_err, "hybrid_ms": hybrid_ms,
        }

    return results


def print_hybrid_table(results, seq_lens):
    print(f"\n{'T':>8}  {'brand err':>10}  {'tkv err':>11}  {'hybrid err':>11}"
          f"  {'brand ms':>9}  {'tkv ms':>10}  {'hybrid ms':>10}")
    print("-" * 80)
    for t in seq_lens:
        r = results[t]
        print(f"{t:>8,}  {r['brand_err']:>10.4f}  {r['tkv_err']:>11.4f}"
              f"  {r['hybrid_err']:>11.4f}  {r['brand_ms']:>9.1f}"
              f"  {r['tkv_ms']:>10.1f}  {r['hybrid_ms']:>10.1f}")


def plot_hybrid(results, seq_lens, path="benchmark_hybrid.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(seq_lens, [results[t]["brand_err"] for t in seq_lens], marker="o", label="BrandOnly")
    ax.plot(seq_lens, [results[t]["tkv_err"] for t in seq_lens], marker="s", linestyle="--", label="TKV only")
    ax.plot(seq_lens, [results[t]["hybrid_err"] for t in seq_lens], marker="^", linestyle="-.", label="Hybrid (ours)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Hybrid vs BrandOnly vs TKV: Reconstruction Error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(seq_lens, [results[t]["brand_ms"] for t in seq_lens], marker="o", label="BrandOnly")
    ax.plot(seq_lens, [results[t]["tkv_ms"] for t in seq_lens], marker="s", linestyle="--", label="TKV only")
    ax.plot(seq_lens, [results[t]["hybrid_ms"] for t in seq_lens], marker="^", linestyle="-.", label="Hybrid (ours)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Total update time (ms)")
    ax.set_title("Hybrid vs BrandOnly vs TKV: Update Cost")
    ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def benchmark_tkv_vs_oja_memory(
    seq_lens=(1_000, 8_000, 32_000, 128_000),
    pyramid_ranks=None,
    uniform_rank=8,
    d=64,
):
    """
    Theoretical memory comparison across methods at long context lengths.
    Reports GB for full KV, uniform-rank OjaKV-style, and TKV pyramid.
    """
    if pyramid_ranks is None:
        pyramid_ranks = PYRAMID_RANKS_GPT2

    results = {}
    for seq_len in seq_lens:
        full_bytes = 4 * 2 * seq_len * d * _GPT2_LAYERS * _GPT2_HEADS  # float32

        oja_bytes = sum(
            _tkv_bytes(seq_len, uniform_rank, d) * _GPT2_HEADS
            for _ in range(_GPT2_LAYERS)
        )

        pyramid_bytes = sum(
            _tkv_bytes(seq_len, pyramid_ranks[l], d) * _GPT2_HEADS
            for l in range(_GPT2_LAYERS)
        )

        results[seq_len] = {
            "full_gb":        full_bytes    / 1e9,
            "oja_uniform_gb": oja_bytes     / 1e9,
            "tkv_pyramid_gb": pyramid_bytes / 1e9,
            "pyramid_vs_full":  full_bytes / pyramid_bytes,
        }

    return results


def print_tkv_memory_table(results, seq_lens):
    print(f"\n{'seq_len':>10}  {'full KV':>9}  {'OjaKV r=8':>10}  {'TKV pyramid':>15}  {'compression':>12}")
    print("-" * 65)
    for t in seq_lens:
        r = results[t]
        print(
            f"{t:>10,}  {r['full_gb']:>8.3f}GB  {r['oja_uniform_gb']:>9.3f}GB"
            f"  {r['tkv_pyramid_gb']:>14.3f}GB  {r['pyramid_vs_full']:>10.1f}x"
        )


def benchmark_tkv_reconstruction(
    seq_lens=(1_000, 8_000, 32_000),
    ranks=(4, 8, 16),
    warmup_len=512,
    d=64,
    seed=0,
):
    """
    Frobenius reconstruction error for TKV (frozen-basis) at long context lengths.
    Uses warmup_len tokens to establish the basis; remaining tokens are projected.
    Compares against OjaKV at the same rank.
    """
    key = jax.random.PRNGKey(seed)
    tkv_err, oja_err = {}, {}

    for seq_len in seq_lens:
        k1, k2 = jax.random.split(key)
        decay = jnp.array([1.0 / (i + 1) ** 0.5 for i in range(d)])
        K_true = jax.random.normal(k1, (seq_len, d)) * decay
        V_true = jax.random.normal(k2, (seq_len, d))
        ref = float(jnp.linalg.norm(K_true)) + 1e-9

        for rank in ranks:
            # TKV (frozen basis)
            state_tkv = tkv_cold_start(K_true, V_true, rank, warmup_len=warmup_len)
            K_tkv, _ = tkv_reconstruct(state_tkv)
            tkv_err[(seq_len, rank)] = float(jnp.linalg.norm(K_true - K_tkv)) / ref

            # OjaKV (streaming, same rank)
            T_max = seq_len + 1
            state_o = init_oja_kv(T_max, rank, d)
            for i in range(seq_len):
                state_o = append_token_oja(state_o, K_true[i], V_true[i], rank)
            K_oj, _ = reconstruct_kv_oja(state_o, T_max)
            oja_err[(seq_len, rank)] = float(jnp.linalg.norm(K_true - K_oj[:seq_len])) / ref

    return tkv_err, oja_err


def print_tkv_reconstruction_table(tkv_err, oja_err, seq_lens, ranks):
    print(f"\n{'seq_len':>10}  {'rank':>5}  {'tkv err':>11}  {'oja err':>10}  {'tkv wins':>12}")
    print("-" * 60)
    for seq_len in seq_lens:
        for rank in ranks:
            te = tkv_err[(seq_len, rank)]
            oe = oja_err[(seq_len, rank)]
            print(f"{seq_len:>10,}  {rank:>5}  {te:>11.4f}  {oe:>10.4f}  {'yes' if te <= oe else 'no':>12}")


def plot_tkv_reconstruction(tkv_err, oja_err, seq_lens, ranks, path="benchmark_tkv_reconstruction.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for rank in ranks:
        te = [tkv_err[(t, rank)] for t in seq_lens]
        oe = [oja_err[(t, rank)] for t in seq_lens]
        ax.plot(seq_lens, te, marker="o", color="tomato", label=f"TKV r={rank}")
        ax.plot(seq_lens, oe, marker="s", linestyle="--", color="seagreen", label=f"OjaKV r={rank}")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("TKV vs OjaKV: Reconstruction Error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=7)

    ax = axes[1]
    for seq_len in seq_lens:
        te = [tkv_err[(seq_len, r)] for r in ranks]
        ax.plot(ranks, te, marker="o", label=f"t={seq_len:,}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("TKV Error vs Rank")
    ax.set_yscale("log")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def benchmark_tkv_perplexity(
    pyramid_ranks=None,
    max_length=128,
    device="cpu",
):
    """
    Perplexity comparison: exact GPT-2 vs BrandOnly (rank=16) vs TKV pyramid.
    """
    from model_hooks import (
        load_gpt2, compute_perplexity,
        patch_gpt2_for_compression, unpatch_gpt2,
        patch_gpt2_tkv, make_phi_for_gpt2, PYRAMID_RANKS,
        patch_model_oja, unpatch_model,
    )
    if pyramid_ranks is None:
        pyramid_ranks = PYRAMID_RANKS

    print("Loading GPT-2...")
    model, tokenizer = load_gpt2(device)

    ppl_exact = compute_perplexity(model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device)
    print(f"  Exact perplexity:                      {ppl_exact:.2f}")

    orig = patch_gpt2_tkv(model, pyramid_ranks)
    ppl_tkv = compute_perplexity(model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device)
    unpatch_gpt2(model, orig)
    avg_rank = sum(pyramid_ranks) // len(pyramid_ranks)
    print(f"  TKV pyramid (avg_rank={avg_rank}):           {ppl_tkv:.2f}  ratio={ppl_tkv/ppl_exact:.4f}")

    phi = make_phi_for_gpt2(16, 64)
    orig = patch_gpt2_for_compression(model, phi, 16, 512)
    ppl_brand = compute_perplexity(model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device)
    unpatch_gpt2(model, orig)
    print(f"  BrandOnly (uniform rank=16):           {ppl_brand:.2f}  ratio={ppl_brand/ppl_exact:.4f}")

    orig = patch_model_oja(model, pyramid_ranks)
    ppl_oja = compute_perplexity(model, tokenizer, SAMPLE_TEXTS, max_length=max_length, device=device)
    unpatch_model(model, orig)
    print(f"  OjaKV (pyramid):                       {ppl_oja:.2f}  ratio={ppl_oja/ppl_exact:.4f}")

    return {"exact": ppl_exact, "tkv_pyramid": ppl_tkv, "brand_r16": ppl_brand, "oja": ppl_oja}


# ---------------------------------------------------------------------------
# Sinked benchmark: full-precision sink+recent vs plain TKV
# ---------------------------------------------------------------------------

def benchmark_sinked(
    seq_lens=(256, 1_000, 4_000),
    rank=8,
    sink_len=64,
    window_len=64,
    warmup_len=512,
    d=64,
    seed=0,
):
    """
    Compares three methods at each seq_len:
      - tkv_only: TKV (frozen-basis), all tokens compressed
      - sinked:   full-precision sink+recent, TKV middle (StreamingLLM + ours)

    Also computes exact attention as the quality ceiling, and reports:
      - Reconstruction error on sink tokens (should be ~0 for sinked)
      - Reconstruction error on the full sequence
      - Memory overhead of full-precision buffers
    """
    key = jax.random.PRNGKey(seed)
    results = {}

    for seq_len in seq_lens:
        k1, k2, k3 = jax.random.split(key, 3)
        decay = jnp.array([1.0 / (i + 1) ** 0.5 for i in range(d)])
        K_true = jax.random.normal(k1, (seq_len, d)) * decay
        V_true = jax.random.normal(k2, (seq_len, d))
        Q      = jax.random.normal(k3, (seq_len, d)) * 0.1
        ref    = float(jnp.linalg.norm(K_true)) + 1e-9
        n_sink = min(sink_len, seq_len)

        # TKV only
        state_f  = tkv_cold_start(K_true, V_true, rank, warmup_len=warmup_len)
        K_f, _   = tkv_reconstruct(state_f)
        err_tkv_sink  = float(jnp.linalg.norm(K_true[:n_sink] - K_f[:n_sink])) / ref
        err_tkv_total = float(jnp.linalg.norm(K_true - K_f)) / ref
        out_f = tkv_compressed_attention(Q, state_f, causal=True)

        # Sinked (sink + recent full precision, TKV middle)
        state_s  = sinked_cold_start(K_true, V_true, rank,
                                     sink_len=sink_len, window_len=window_len,
                                     warmup_len=warmup_len)
        # sink reconstruction error is exactly 0 by construction
        err_s_sink  = 0.0
        T_mid = state_s.middle.seq_len if state_s.middle else 0
        if T_mid > 0:
            assert state_s.middle is not None
            K_mid, _ = tkv_reconstruct(state_s.middle)
            K_approx  = jnp.concatenate([
                jnp.array(state_s.sink_k),
                K_mid,
                jnp.array(state_s.recent_k),
            ])
        else:
            K_approx = jnp.concatenate([
                jnp.array(state_s.sink_k),
                jnp.array(state_s.recent_k),
            ])
        err_s_total = float(jnp.linalg.norm(K_true - K_approx)) / ref
        out_s = sinked_attention(Q, state_s, causal=True)

        # Attention quality: relative output error vs plain TKV
        attn_err = float(jnp.linalg.norm(out_s - out_f) / (jnp.linalg.norm(out_f) + 1e-9))

        # Memory: sinked adds 2 * (sink_len + window_len) * d full-precision floats per head
        extra_fp_mb = 4 * 2 * (n_sink + min(window_len, seq_len)) * d / 1e6

        results[seq_len] = {
            "err_tkv_sink":     err_tkv_sink,
            "err_sinked_sink":  err_s_sink,
            "err_tkv_total":    err_tkv_total,
            "err_sinked_total": err_s_total,
            "attn_err_vs_tkv":  attn_err,
            "extra_fp_mb":      extra_fp_mb,
        }

    return results


def print_sinked_table(results, seq_lens):
    print(f"\n{'T':>8}  {'tkv sink err':>16}  {'sinked sink err':>16}"
          f"  {'tkv total':>13}  {'sinked total':>13}  {'attn diff':>10}  {'extra MB':>9}")
    print("-" * 95)
    for t in seq_lens:
        r = results[t]
        print(
            f"{t:>8,}  {r['err_tkv_sink']:>16.4f}  {r['err_sinked_sink']:>16.4f}"
            f"  {r['err_tkv_total']:>13.4f}  {r['err_sinked_total']:>13.4f}"
            f"  {r['attn_err_vs_tkv']:>10.4f}  {r['extra_fp_mb']:>8.3f}MB"
        )


def plot_sinked(results, seq_lens, path="benchmark_sinked.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(seq_lens, [results[t]["err_tkv_total"] for t in seq_lens],
            marker="o", label="TKV only")
    ax.plot(seq_lens, [results[t]["err_sinked_total"] for t in seq_lens],
            marker="s", linestyle="--", label="sinked (ours)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Total Reconstruction Error: TKV vs Sinked")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(seq_lens, [results[t]["attn_err_vs_tkv"] for t in seq_lens],
            marker="^", color="tomato")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Attention Output Error")
    ax.set_title("Sinked Attention Quality vs TKV Basis")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Metric 11: TKV streaming decode vs OjaKV — per-token update time
# ---------------------------------------------------------------------------

def benchmark_tkv_streaming_vs_oja(
    warmup_len: int = 256,
    decode_len: int = 256,
    ranks=(8, 16, 32),
    d: int = 64,
    T_max: int = 2048,
):
    """
    Measures decode-phase per-token update time after a shared warmup.

    Warmup: first warmup_len tokens processed by Brand (BrandOnly) or OjaKV.
    Decode: next decode_len tokens — only this phase is timed.

    TKV uses init_tkv_streaming to pre-allocate the proj buffer,
    then tkv_add_token writes in-place (O(R·d), no copy).  OjaKV still
    runs QR on every decode token (O(R²·d)).

    Uses real K/V from GPT-2 small (layer 6, head 0) on WikiText-2.
    """
    print("Extracting real K/V from GPT-2...")
    K_real, V_real = extract_real_kv(max_seq_len=warmup_len + decode_len)
    K_warm, V_warm   = K_real[:warmup_len],   V_real[:warmup_len]
    K_decode, V_decode = K_real[warmup_len:], V_real[warmup_len:]
    K_all = jnp.concatenate([K_warm, K_decode], axis=0)

    hdr = f"{'rank':>4}  {'BrandOnly':>12}  {'TKV':>12}  {'OjaKV':>12}  tkv_err  oja_err"
    print(f"\nTKV Streaming vs OjaKV — warmup={warmup_len}, decode={decode_len} tokens")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for rank in ranks:
        # --- BrandOnly (decode phase only) ---
        state_e = init_compressed_kv(T_max, rank, d)
        for i in range(warmup_len):
            state_e = append_token(state_e, K_warm[i], V_warm[i], rank)
        t0 = time.perf_counter()
        for i in range(decode_len):
            state_e = append_token(state_e, K_decode[i], V_decode[i], rank)
        jax.block_until_ready(state_e.U_k)
        ms_brand = (time.perf_counter() - t0) / decode_len * 1e3

        # --- TKV streaming (decode phase only) ---
        state_b = init_compressed_kv(T_max, rank, d)
        for i in range(warmup_len):
            state_b = append_token(state_b, K_warm[i], V_warm[i], rank)
        state_tkv = init_tkv_streaming(state_b, T_max)
        t0 = time.perf_counter()
        for i in range(decode_len):
            state_tkv = tkv_add_token(state_tkv, K_decode[i], V_decode[i])
        ms_tkv = (time.perf_counter() - t0) / decode_len * 1e3
        K_trec, _ = tkv_reconstruct(state_tkv)
        tkv_err = float(
            jnp.linalg.norm(K_all - K_trec[:warmup_len + decode_len])
            / (jnp.linalg.norm(K_all) + 1e-9)
        )

        # --- OjaKV (decode phase only) ---
        state_o = init_oja_kv(T_max, rank, d)
        for i in range(warmup_len):
            state_o = append_token_oja(state_o, K_warm[i], V_warm[i], rank)
        t0 = time.perf_counter()
        for i in range(decode_len):
            state_o = append_token_oja(state_o, K_decode[i], V_decode[i], rank)
        jax.block_until_ready(state_o.V_k)
        ms_oja = (time.perf_counter() - t0) / decode_len * 1e3
        K_orec, _ = reconstruct_kv_oja(state_o, T_max)
        oja_err = float(
            jnp.linalg.norm(K_all - K_orec[:warmup_len + decode_len])
            / (jnp.linalg.norm(K_all) + 1e-9)
        )

        print(
            f"{rank:4d}  {ms_brand:>10.3f}ms  {ms_tkv:>10.3f}ms  {ms_oja:>10.3f}ms"
            f"  {tkv_err:.4f}      {oja_err:.4f}"
        )
        results[rank] = {
            "ms_brand": ms_brand, "ms_tkv": ms_tkv, "ms_oja": ms_oja,
            "tkv_err": tkv_err, "oja_err": oja_err,
        }

    return results


def plot_tkv_streaming_vs_oja(results, ranks, path="benchmark_tkv_streaming.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(ranks, [results[r]["ms_tkv"] for r in ranks], marker="s", linestyle="--", color="tomato", label="TKV")
    ax.plot(ranks, [results[r]["ms_oja"]    for r in ranks], marker="^", linestyle="-.", color="seagreen", label="OjaKV")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Per-token update time (ms)")
    ax.set_title("Decode-Phase Update Cost vs Rank")
    ax.legend()

    ax = axes[1]
    ax.plot(ranks, [results[r]["tkv_err"] for r in ranks], marker="s", linestyle="--", color="tomato", label="TKV")
    ax.plot(ranks, [results[r]["oja_err"]    for r in ranks], marker="^", linestyle="-.", color="seagreen", label="OjaKV")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Decode-Phase Reconstruction Error vs Rank")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Summary: TKV vs OjaKV head-to-head
# ---------------------------------------------------------------------------

def plot_head_to_head(ppl_results, streaming_results, tkv_err, oja_err,
                      path="benchmark_head_to_head.pdf"):
    """
    3-panel summary comparing TKV (ours) vs OjaKV:
      Left:   per-token decode time vs rank
      Center: reconstruction error vs context length (rank=16)
      Right:  GPT-2 perplexity bar chart (exact / tkv / oja)
    """
    ranks = sorted(streaming_results.keys())
    seq_lens = sorted({sl for sl, _ in tkv_err.keys()})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: per-token update time vs rank
    ax = axes[0]
    ax.plot(ranks, [streaming_results[r]["ms_tkv"] for r in ranks],
            marker="s", linestyle="--", color="tomato", label="TKV")
    ax.plot(ranks, [streaming_results[r]["ms_oja"] for r in ranks],
            marker="^", linestyle="-.", color="seagreen", label="OjaKV")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Per-token update time (ms)")
    ax.set_title("Decode Speed vs Rank")
    ax.legend()

    # Center: reconstruction error vs context length at rank=16
    ax = axes[1]
    ax.plot(seq_lens, [tkv_err[(sl, 16)] for sl in seq_lens],
            marker="s", linestyle="--", color="tomato", label="TKV")
    ax.plot(seq_lens, [oja_err[(sl, 16)] for sl in seq_lens],
            marker="^", linestyle="-.", color="seagreen", label="OjaKV")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Reconstruction Error at Rank=16")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # Right: GPT-2 perplexity bar chart (TKV vs OjaKV + exact baseline)
    ax = axes[2]
    labels = ["Exact", "TKV", "OjaKV"]
    values = [ppl_results["exact"], ppl_results["tkv_pyramid"], ppl_results["oja"]]
    colors = ["steelblue", "tomato", "seagreen"]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("GPT-2 Perplexity Comparison")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEQ_LENS = (32, 64, 128, 256, 512)
    RANKS = (2, 4, 8, 16, 32)
    D = 64
    T_MAX = 1024

    print("=" * 50)
    print("Metric 1: Reconstruction Error")
    print("=" * 50)
    recon = benchmark_reconstruction(SEQ_LENS, RANKS, D, T_MAX)
    plot_reconstruction(recon, SEQ_LENS, RANKS)
    for rank in RANKS:
        err = recon[(512, rank)]
        print(f"  t=512, r={rank}: rel_err={err:.4f}")

    print()
    print("=" * 50)
    print("Metric 2: Memory Compression Ratio")
    print("=" * 50)
    mem = benchmark_memory(SEQ_LENS, RANKS, D)
    plot_memory(mem, SEQ_LENS, RANKS)

    print()
    print("=" * 50)
    print("Metric 3: Perplexity on GPT-2")
    print("=" * 50)
    ppl_results = benchmark_perplexity(
        ranks=(4, 8, 16, 32, 48),
        max_length=128,
        T_max=T_MAX,
        head_dim=D,
    )
    plot_perplexity(ppl_results)

    print()
    print("=" * 50)
    print("Metric 4: BrandOnly vs OjaKV Head-to-Head")
    print("=" * 50)
    brand_err, oja_err, brand_time, oja_time = benchmark_brand_vs_oja(SEQ_LENS, RANKS, D, T_MAX)
    brand_curve, oja_curve = benchmark_streaming_error(rank=16, seq_len=512, d=D, T_max=T_MAX)
    plot_comparison(brand_err, oja_err, brand_time, oja_time, brand_curve, oja_curve, SEQ_LENS, RANKS)
    print_comparison_table(brand_err, oja_err, brand_time, oja_time, SEQ_LENS, RANKS)

    print()
    print("=" * 50)
    print("Metric 4b: Multi-Layer/Head Comparison (BrandOnly vs OjaKV)")
    print("=" * 50)
    benchmark_multi_head_comparison(
        layers=(0, 3, 6, 9, 11),
        heads=(0, 4, 8),
        seq_lens=(64, 128, 256, 512),
        ranks=(8, 16, 32),
        d=D,
        T_max=T_MAX,
    )

    print()
    print("=" * 50)
    print("Metric 5: Lazy Flush Sweep (BrandOnly vs OjaKV)")
    print("=" * 50)
    FLUSH_VALS = (1, 2, 4, 8)
    flush_results = benchmark_flush_sweep(
        flush_every_vals=FLUSH_VALS,
        seq_lens=(64, 256, 512),
        ranks=(8, 16, 32),
        d=D,
        T_max=T_MAX,
    )
    plot_flush_sweep(flush_results, FLUSH_VALS, (64, 256, 512), (8, 16, 32))
    print_flush_table(flush_results, FLUSH_VALS, seq_len=512, ranks=(8, 16, 32))

    print()
    print("=" * 50)
    print("Metric 6: Importance-Weighted Cold Start")
    print("=" * 50)
    imp_results = benchmark_importance(ranks=(4, 8, 16, 32), seq_len=128, d=D, T_max=T_MAX)
    plot_importance(imp_results, ranks=(4, 8, 16, 32))
    print(f"  {'rank':>4}  {'imp_uni':>8}  {'imp_wt':>8}  {'gain':>6}")
    for r in (4, 8, 16, 32):
        u = imp_results[r]["err_imp_uni"]
        w = imp_results[r]["err_imp_wt"]
        print(f"  {r:>4}  {u:>8.4f}  {w:>8.4f}  {(u-w)/u*100:>5.1f}%")

    print()
    print("=" * 50)
    print("Metric 7: Hybrid Streaming (Brand warmup → TKV long-context)")
    print("=" * 50)
    HYBRID_SEQ_LENS = (128, 512, 1_000, 4_000, 16_000)
    hybrid_results = benchmark_hybrid_streaming(
        seq_lens=HYBRID_SEQ_LENS, rank=16, warmup_len=512, d=D
    )
    print_hybrid_table(hybrid_results, HYBRID_SEQ_LENS)
    plot_hybrid(hybrid_results, HYBRID_SEQ_LENS)

    print()
    print("=" * 50)
    print("Metric 8: TKV Memory vs OjaKV at 128K+")
    print("=" * 50)
    LONG_SEQ_LENS = (1_000, 8_000, 32_000, 128_000)
    mem_tkv = benchmark_tkv_vs_oja_memory(LONG_SEQ_LENS, PYRAMID_RANKS_GPT2, uniform_rank=8, d=D)
    print_tkv_memory_table(mem_tkv, LONG_SEQ_LENS)

    print()
    print("=" * 50)
    print("Metric 8b: TKV Reconstruction Error at Long Context")
    print("=" * 50)
    LONG_SEQ_LENS_ERR = (1_000, 8_000, 32_000)
    tkv_err_long, oja_err_long = benchmark_tkv_reconstruction(
        seq_lens=LONG_SEQ_LENS_ERR, ranks=(4, 8, 16), warmup_len=512, d=D
    )
    print_tkv_reconstruction_table(tkv_err_long, oja_err_long, LONG_SEQ_LENS_ERR, ranks=(4, 8, 16))
    plot_tkv_reconstruction(tkv_err_long, oja_err_long, LONG_SEQ_LENS_ERR, ranks=(4, 8, 16))

    print()
    print("=" * 50)
    print("Metric 9: TKV Perplexity on GPT-2")
    print("=" * 50)
    ppl_tkv_results = benchmark_tkv_perplexity(pyramid_ranks=PYRAMID_RANKS_GPT2, max_length=128)

    print()
    print("=" * 50)
    print("Metric 10: Sinked (full-precision sink+recent) vs TKV")
    print("=" * 50)
    sinked_results = benchmark_sinked(
        seq_lens=(256, 1_000, 4_000), rank=8,
        sink_len=64, window_len=64, warmup_len=512, d=D,
    )
    print_sinked_table(sinked_results, (256, 1_000, 4_000))
    plot_sinked(sinked_results, (256, 1_000, 4_000))

    print()
    print("=" * 50)
    print("Metric 11: TKV Streaming vs OjaKV (Decode Phase)")
    print("=" * 50)
    streaming_results = benchmark_tkv_streaming_vs_oja(
        warmup_len=256, decode_len=256, ranks=(8, 16, 32), d=D, T_max=T_MAX,
    )
    plot_tkv_streaming_vs_oja(streaming_results, ranks=(8, 16, 32))

    print()
    print("=" * 50)
    print("Summary: TKV vs OjaKV Head-to-Head")
    print("=" * 50)
    plot_head_to_head(ppl_tkv_results, streaming_results, tkv_err_long, oja_err_long)

    print()
    print("All benchmarks complete.")
