"""
Perplexity-vs-rank sweep across model scales and context lengths.

Answers: does the low-rank structure in attention KV caches persist as
model scale (GPT-2 → TinyLlama-1.1B) and context length increase?

Method: TKV (warmup-then-freeze) SVD compression with flat rank across all layers.
SVD computed on the full context window (optimal basis for that sequence),
so rank is the sole variable — no warmup truncation artefact.

Run:
  python rank_sweep_benchmark.py                        # GPT-2 only (fast)
  python rank_sweep_benchmark.py --models tinyllama     # adds TinyLlama
  python rank_sweep_benchmark.py --models gpt2 tinyllama --ctx-lens 128 512 1024
"""

import argparse
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List

from wikitext_benchmark import load_wikitext2, sliding_perplexity
from model_hooks import load_model, load_gpt2, patch_model_tkv_flat, unpatch_model

MODEL_SPECS = {
    "gpt2": {
        "hf_name": "gpt2",
        "label": "GPT-2 (117M)",
        "color": "steelblue",
        "head_dim": 64,
        "dtype": torch.float32,
    },
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama-1.1B",
        "color": "tomato",
        "head_dim": 64,
        "dtype": torch.bfloat16,
    },
}

RANKS = [4, 8, 16, 32, 48]
CTX_LENS = [128, 512, 1024]


def sweep_one_model(
    model_key: str,
    ranks: List[int],
    ctx_lens: List[int],
    device: str = "cpu",
    max_seqs: int = 8,
    stride: int = 128,
) -> Dict:
    """
    Returns results[ctx_len] = {
        "exact": ppl,
        rank_int: ppl,
        ...
    }
    """
    spec = MODEL_SPECS[model_key]
    print(f"\n{'='*60}")
    print(f"Model: {spec['label']}")
    print(f"{'='*60}")

    if model_key == "gpt2":
        model, tok = load_gpt2(device)
    else:
        model, tok = load_model(spec["hf_name"], device=device, dtype=spec["dtype"])

    print("Loading WikiText-2...")
    text = load_wikitext2(n_tokens=60000)

    results = {}

    for ctx in ctx_lens:
        print(f"\n  ctx_len = {ctx}")
        results[ctx] = {}

        # Baseline: exact attention
        ppl_exact = sliding_perplexity(model, tok, text, ctx, stride, device, max_seqs)
        results[ctx]["exact"] = ppl_exact
        print(f"    exact:    {ppl_exact:.2f}")

        # Rank sweep
        for rank in ranks:
            # Use a unique registry key per (model, rank) to avoid stale closures
            key = f"tkv_flat_r{rank}"
            orig = patch_model_tkv_flat(model, rank, key=key)
            try:
                ppl = sliding_perplexity(model, tok, text, ctx, stride, device, max_seqs)
            finally:
                unpatch_model(model, orig)

            results[ctx][rank] = ppl
            ratio = ppl / ppl_exact
            print(f"    rank={rank:2d}:  {ppl:.2f}  (ratio {ratio:.3f}x)")

    return results


def print_table(all_results: Dict, ranks: List[int], ctx_lens: List[int]):
    """Print perplexity degradation ratio table: rows=rank, cols=ctx_len × model."""
    print("\n" + "="*70)
    print("Perplexity ratio (compressed / exact) — lower is better, 1.00 = lossless")
    print("="*70)

    models = list(all_results.keys())
    header = f"{'rank':>5}"
    for ctx in ctx_lens:
        for m in models:
            label = MODEL_SPECS[m]["label"].split()[0]
            header += f"  {label}@{ctx}"
    print(header)
    print("-"*len(header))

    for rank in ranks:
        row = f"{rank:>5}"
        for ctx in ctx_lens:
            for m in models:
                res = all_results[m].get(ctx, {})
                exact = res.get("exact", float("nan"))
                ppl = res.get(rank, float("nan"))
                ratio = ppl / exact if not (math.isnan(ppl) or math.isnan(exact)) else float("nan")
                row += f"  {ratio:>9.3f}" if not math.isnan(ratio) else f"  {'N/A':>9}"
        print(row)
    print("="*70)


def plot_sweep(all_results: Dict, ranks: List[int], ctx_lens: List[int],
               path: str = "rank_sweep.pdf"):
    """
    One subplot per context length.  Each subplot shows perplexity ratio
    vs rank for each model.  A ratio of 1.0 = lossless.
    """
    n_ctx = len(ctx_lens)
    fig, axes = plt.subplots(1, n_ctx, figsize=(5 * n_ctx, 4), sharey=False)
    if n_ctx == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctx_lens):
        for model_key, res in all_results.items():
            spec = MODEL_SPECS[model_key]
            ctx_res = res.get(ctx, {})
            exact = ctx_res.get("exact", float("nan"))
            if math.isnan(exact):
                continue
            ratios = [ctx_res.get(r, float("nan")) / exact for r in ranks]
            ax.plot(ranks, ratios, marker="o", color=spec["color"], label=spec["label"])

        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="lossless")
        ax.set_xlabel("Rank (flat across all layers)")
        ax.set_ylabel("Perplexity ratio (compressed / exact)")
        ax.set_title(f"Context length = {ctx} tokens")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.9)

    fig.suptitle("Perplexity vs Rank: Does Low-Rank Structure Persist Across Model Scales?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt2"],
                        choices=list(MODEL_SPECS.keys()),
                        help="Models to benchmark (default: gpt2)")
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=CTX_LENS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seqs", type=int, default=8,
                        help="WikiText windows per (model, ctx_len, rank) — keep <=10 for speed")
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()

    all_results = {}
    for model_key in args.models:
        all_results[model_key] = sweep_one_model(
            model_key,
            ranks=args.ranks,
            ctx_lens=args.ctx_lens,
            device=args.device,
            max_seqs=args.max_seqs,
            stride=args.stride,
        )

    print_table(all_results, args.ranks, args.ctx_lens)
    plot_sweep(all_results, args.ranks, args.ctx_lens)
