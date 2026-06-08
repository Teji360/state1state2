"""
Needle-in-a-Haystack benchmark for KV cache compression methods.

Inserts a single secret word into a long filler context at various depths,
then asks the model to recall it. Measures retrieval probability (P(correct token))
as a function of context length and compression method.

Mirrors the RULER NIAH evaluation used in the OjaKV paper (arXiv:2509.21623).
"""

import argparse
import math
import random
import sys
import torch
from typing import List, Tuple

from model_hooks import (
    load_model,
    patch_model_sinked,
    patch_model_oja,
    unpatch_model,
    _default_pyramid,
)

# ---------------------------------------------------------------------------
# Needle design
# ---------------------------------------------------------------------------

# Single-token words in LLaMA/Mistral tokenizers — verified empirically.
NEEDLE_WORDS = ["blue", "gold", "dark", "soft", "tall", "wild", "cold", "warm"]

FILLER = (
    "In the field of natural language processing, researchers have made significant "
    "advances in understanding how language models process and generate text. "
    "These models learn statistical patterns from large corpora and use this knowledge "
    "to predict the most likely next token in a sequence. The architecture relies on "
    "self-attention mechanisms that allow each token to interact with all previous tokens. "
    "Modern large language models contain billions of parameters and are trained on "
    "trillions of tokens of web text, books, and code. "
)

NEEDLE_TEMPLATE = "The special code word is {word}. Remember it carefully. "
QUESTION_TEMPLATE = "Based on the passage above, what is the special code word? The special code word is"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    tokenizer,
    target_len: int,
    needle_word: str,
    needle_depth: float = 0.5,
    max_context: int = 131072,
) -> Tuple[List[int], int]:
    """
    Returns (input_ids, needle_token_id).

    Fills to approximately target_len tokens:
      [filler × A] [needle sentence] [filler × B] [question]
    Needle is placed at needle_depth * target_len from the start.
    """
    needle_str = NEEDLE_TEMPLATE.format(word=needle_word)
    question_str = QUESTION_TEMPLATE

    needle_ids = tokenizer.encode(needle_str, add_special_tokens=False)
    question_ids = tokenizer.encode(question_str, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)

    # Tokens reserved for needle + question
    reserved = len(needle_ids) + len(question_ids)
    filler_budget = max(0, target_len - reserved)

    needle_pos_tokens = int(filler_budget * needle_depth)
    before_tokens = needle_pos_tokens
    after_tokens = filler_budget - before_tokens

    def tile(ids, n):
        out = []
        while len(out) < n:
            out.extend(ids)
        return out[:n]

    before = tile(filler_ids, before_tokens)
    after  = tile(filler_ids, after_tokens)

    # BOS token if the tokenizer uses one
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    input_ids = bos + before + needle_ids + after + question_ids

    # Clip to model's max context
    if len(input_ids) > max_context:
        input_ids = input_ids[:max_context]

    # Token id of the needle word
    word_ids = tokenizer.encode(" " + needle_word, add_special_tokens=False)
    if not word_ids:
        word_ids = tokenizer.encode(needle_word, add_special_tokens=False)
    needle_token_id = word_ids[0]

    return input_ids, needle_token_id


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_retrieval(
    model,
    input_ids: List[int],
    needle_token_id: int,
    device: str,
) -> float:
    """
    Returns P(needle_token | context) — the softmax probability of the correct
    next token immediately after the question prompt.
    """
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits[0, -1]          # (vocab_size,)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[needle_token_id].item()


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_needle_sweep(
    model_name: str,
    context_lens: List[int],
    n_trials: int = 5,
    sink_len: int = 64,
    window_len: int = 64,
    warmup_len: int = 512,
    needle_depth: float = 0.5,
    device: str = "cpu",
    dtype=torch.bfloat16,
    oja_T_max_cap: int = 32768,
) -> dict:
    """
    Runs needle retrieval at each context length for three methods:
      - full_kv:  exact attention (baseline)
      - sinked:   our frozen-basis + sink+recent (proposed)
      - oja:      bare Oja rule (OjaKV equivalent baseline)

    Returns nested dict: results[method][context_len] = mean retrieval probability.
    """
    print(f"\nLoading {model_name}...")
    model, tok = load_model(model_name, device=device, dtype=dtype)
    n_layers = model.config.num_hidden_layers
    pyramid_ranks = _default_pyramid(n_layers)

    max_ctx = getattr(model.config, "max_position_embeddings", 131072)
    print(f"  layers={n_layers}  max_ctx={max_ctx}  pyramid_avg={sum(pyramid_ranks)/len(pyramid_ranks):.1f}")

    results = {m: {} for m in ("full_kv", "sinked", "oja")}
    words = NEEDLE_WORDS

    for ctx_len in context_lens:
        if ctx_len > max_ctx:
            print(f"  Skipping ctx={ctx_len:,} — exceeds model max_position_embeddings {max_ctx}")
            for m in results:
                results[m][ctx_len] = None
            continue

        scores = {m: [] for m in results}
        for trial in range(n_trials):
            word = words[trial % len(words)]
            input_ids, needle_tok = build_prompt(tok, ctx_len, word, needle_depth, max_ctx)
            actual_len = len(input_ids)

            # Full KV
            try:
                p = eval_retrieval(model, input_ids, needle_tok, device)
                scores["full_kv"].append(p)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    scores["full_kv"].append(float("nan"))
                    print(f"    full_kv OOM at ctx={ctx_len:,}")
                else:
                    raise

            # Sinked (our method)
            orig = patch_model_sinked(model, sink_len, window_len, warmup_len, pyramid_ranks)
            try:
                p = eval_retrieval(model, input_ids, needle_tok, device)
                scores["sinked"].append(p)
            except RuntimeError as e:
                scores["sinked"].append(float("nan"))
                if "out of memory" in str(e).lower():
                    print(f"    sinked OOM at ctx={ctx_len:,}")
                else:
                    raise
            finally:
                unpatch_model(model, orig)

            # OjaKV baseline
            orig = patch_model_oja(model, pyramid_ranks)
            try:
                p = eval_retrieval(model, input_ids, needle_tok, device)
                scores["oja"].append(p)
            except RuntimeError as e:
                scores["oja"].append(float("nan"))
                if "out of memory" in str(e).lower():
                    print(f"    oja OOM at ctx={ctx_len:,}")
                else:
                    raise
            finally:
                unpatch_model(model, orig)

            print(f"  ctx={actual_len:6,}  trial={trial+1}/{n_trials}  word={word!r:6}"
                  f"  full={scores['full_kv'][-1]:.4f}"
                  f"  sinked={scores['sinked'][-1]:.4f}"
                  f"  oja={scores['oja'][-1]:.4f}")

        def nanmean(xs):
            valid = [x for x in xs if not math.isnan(x)]
            return sum(valid) / len(valid) if valid else float("nan")

        for m in results:
            results[m][ctx_len] = nanmean(scores[m])

    return results


def print_results(results: dict, context_lens: List[int]):
    methods = list(results.keys())
    header = f"{'ctx_len':>10}" + "".join(f"  {m:>12}" for m in methods)
    print("\n" + "=" * len(header))
    print("Needle-in-a-Haystack Retrieval Probability (higher = better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for ctx in context_lens:
        row = f"{ctx:>10,}"
        for m in methods:
            v = results[m].get(ctx)
            row += f"  {v:>12.4f}" if v is not None and not math.isnan(v) else f"  {'OOM/skip':>12}"
        print(row)
    print("=" * len(header))

    # Summary: sinked vs oja advantage at each context length
    print("\nSinked vs OjaKV delta (sinked − oja, positive = sinked wins):")
    for ctx in context_lens:
        s = results["sinked"].get(ctx)
        o = results["oja"].get(ctx)
        if s is not None and o is not None and not math.isnan(s) and not math.isnan(o):
            delta = s - o
            print(f"  ctx={ctx:>8,}  delta={delta:+.4f}  {'✓ sinked wins' if delta > 0 else '✗ oja wins'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--ctx-lens", nargs="+", type=int,
                        default=[256, 512, 1024, 2048])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sink-len", type=int, default=64)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--warmup-len", type=int, default=256)
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Needle depth as fraction of context (0=start, 1=end)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16
    results = run_needle_sweep(
        model_name=args.model,
        context_lens=args.ctx_lens,
        n_trials=args.trials,
        sink_len=args.sink_len,
        window_len=args.window_len,
        warmup_len=args.warmup_len,
        needle_depth=args.depth,
        device=args.device,
        dtype=dtype,
    )
    print_results(results, args.ctx_lens)
