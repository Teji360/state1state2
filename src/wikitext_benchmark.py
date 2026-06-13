"""
Long-context perplexity benchmark on WikiText-2 for KV cache compression.

Measures sliding-window perplexity at various context lengths for:
  - full_kv:  exact attention (baseline)
  - sinked:   frozen-basis + full-precision sink+recent (proposed)
  - oja:      JAX-scan Oja rule with stale projections (OjaKV baseline)

This mirrors OjaKV's LongBench evaluation (arXiv:2509.21623).
Lower perplexity = better.
"""

import argparse
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import List

from model_hooks import (
    load_model,
    patch_model_sinked,
    patch_model_oja,
    unpatch_model,
    _default_pyramid,
)


def load_wikitext2(n_tokens: int = 50000, split: str = "test") -> str:
    """Load WikiText-2 test set and return a single concatenated string."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return text[:n_tokens * 5]  # rough over-estimate, tokenizer will trim


@torch.no_grad()
def sliding_perplexity(
    model,
    tokenizer,
    text: str,
    context_len: int,
    stride: int = 256,
    device: str = "cpu",
    max_seqs: int = 20,
) -> float:
    """
    Sliding-window perplexity: process chunks of context_len tokens with
    stride overlap. Only count loss on the non-overlapping (stride) tokens
    at the right edge of each window, so every token is predicted in context.

    capped at max_seqs windows to keep runtime manageable.
    """
    ids = tokenizer.encode(text)
    if len(ids) < context_len + stride:
        # pad if text is too short — repeat it
        ids = (ids * ((context_len + stride) // len(ids) + 2))[:context_len + stride * max_seqs + 1]

    total_nll = 0.0
    total_toks = 0
    n_seqs = 0

    for start in range(0, min(len(ids) - context_len, stride * max_seqs), stride):
        end = start + context_len
        chunk = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        try:
            out = model(chunk, use_cache=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return float("nan")
            raise

        # loss on last `stride` tokens only (the newly revealed ones)
        logits = out.logits[0]                     # (T, V)
        target_start = context_len - stride
        log_probs = torch.log_softmax(logits[target_start:-1].float(), dim=-1)
        targets = chunk[0, target_start + 1:]
        nll = -log_probs[range(len(targets)), targets].mean().item()
        total_nll += nll * len(targets)
        total_toks += len(targets)
        n_seqs += 1

    return math.exp(total_nll / total_toks) if total_toks > 0 else float("inf")


def run_ppl_sweep(
    model_name: str,
    context_lens: List[int],
    stride: int = 256,
    sink_len: int = 64,
    window_len: int = 64,
    warmup_len: int = 512,
    device: str = "cpu",
    dtype=torch.bfloat16,
    max_seqs: int = 15,
) -> dict:
    """
    Returns dict[method][context_len] = perplexity.
    """
    print(f"\nLoading {model_name}...")
    model, tok = load_model(model_name, device=device, dtype=dtype)
    n_layers = model.config.num_hidden_layers
    pyramid = _default_pyramid(n_layers)
    print(f"  layers={n_layers}  pyramid_avg={sum(pyramid)/len(pyramid):.1f}")

    print("Loading WikiText-2...")
    text = load_wikitext2(n_tokens=100000)
    ids_total = len(tok.encode(text))
    print(f"  corpus tokens: {ids_total:,}")

    results = {m: {} for m in ("full_kv", "sinked", "oja")}

    for ctx in context_lens:
        print(f"\n  ctx={ctx:,} tokens:")

        # full KV
        try:
            ppl = sliding_perplexity(model, tok, text, ctx, stride, device, max_seqs)
            tag = f"{ppl:.2f}" if not math.isnan(ppl) else "OOM"
        except Exception as e:
            ppl, tag = float("nan"), f"ERR({e})"
        results["full_kv"][ctx] = ppl
        print(f"    full_kv : {tag}")

        # sinked (our method)
        orig = patch_model_sinked(model, sink_len, window_len, warmup_len, pyramid)
        try:
            ppl = sliding_perplexity(model, tok, text, ctx, stride, device, max_seqs)
            tag = f"{ppl:.2f}" if not math.isnan(ppl) else "OOM"
        except Exception as e:
            ppl, tag = float("nan"), f"ERR({e})"
        finally:
            unpatch_model(model, orig)
        results["sinked"][ctx] = ppl
        print(f"    sinked  : {tag}")

        # OjaKV baseline
        orig = patch_model_oja(model, pyramid)
        try:
            ppl = sliding_perplexity(model, tok, text, ctx, stride, device, max_seqs)
            tag = f"{ppl:.2f}" if not math.isnan(ppl) else "OOM"
        except Exception as e:
            ppl, tag = float("nan"), f"ERR({e})"
        finally:
            unpatch_model(model, orig)
        results["oja"][ctx] = ppl
        print(f"    oja     : {tag}")

    return results


def print_ppl_table(results: dict, context_lens: List[int]):
    methods = list(results.keys())
    header = f"\n{'ctx_len':>10}" + "".join(f"  {m:>10}" for m in methods) + "  sinked/full  sinked/oja"
    print("\n" + "=" * (len(header) - 1))
    print("Long-Context Perplexity on WikiText-2 (lower = better)")
    print("=" * (len(header) - 1))
    print(header)
    print("-" * (len(header) - 1))
    for ctx in context_lens:
        row = f"{ctx:>10,}"
        vals = {}
        for m in methods:
            v = results[m].get(ctx, float("nan"))
            vals[m] = v
            row += f"  {v:>10.2f}" if not math.isnan(v) else f"  {'OOM':>10}"
        sf = vals["sinked"] / vals["full_kv"] if not math.isnan(vals["sinked"]) and not math.isnan(vals["full_kv"]) else float("nan")
        so = vals["sinked"] / vals["oja"]     if not math.isnan(vals["sinked"]) and not math.isnan(vals["oja"]) else float("nan")
        row += f"  {sf:>11.3f}" if not math.isnan(sf) else f"  {'N/A':>11}"
        row += f"  {so:>10.3f}" if not math.isnan(so) else f"  {'N/A':>10}"
        print(row)
    print("=" * (len(header) - 1))
    print("sinked/full < 1.10 = <10% perplexity degradation (target)")
    print("sinked/oja  < 1.00 = sinked beats OjaKV (goal)")


def plot_ppl_results(results: dict, context_lens: List[int], path: str = "wikitext_perplexity.pdf"):
    colors  = {"sinked": "tomato", "oja": "seagreen"}
    markers = {"sinked": "s",      "oja": "^"}
    labels  = {"sinked": "FrozenKV", "oja": "OjaKV"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("sinked", "oja"):
        ys = [results[method].get(c, float("nan")) for c in context_lens]
        ax.plot(context_lens, ys, marker=markers[method], color=colors[method], label=labels[method])

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Long-Context Perplexity on WikiText-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--sink-len", type=int, default=64)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--warmup-len", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seqs", type=int, default=15)
    args = parser.parse_args()

    results = run_ppl_sweep(
        model_name=args.model,
        context_lens=args.ctx_lens,
        stride=args.stride,
        sink_len=args.sink_len,
        window_len=args.window_len,
        warmup_len=args.warmup_len,
        device=args.device,
        max_seqs=args.max_seqs,
    )
    print_ppl_table(results, args.ctx_lens)
    plot_ppl_results(results, args.ctx_lens)
