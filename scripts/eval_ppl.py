import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import torch

from pyramidsinkkv import (
    PyramidSinkKVConfig,
    cache_seq_length,
    compress_past_key_values,
    decode_attention_mask_for_cache,
    load_model_and_tokenizer,
    normalize_budget_mode,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate continuation PPL with optional PyramidSinkKV compression.")
    parser.add_argument("--model_name_or_path", default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_length", type=int, default=1024, help="Prefill context length.")
    parser.add_argument("--stride", type=int, default=128, help="Number of continuation tokens scored per window.")
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument(
        "--budget_mode",
        choices=["no_cache", "dense", "uniform", "pyramid", "reversed", "spindle", "hourglass"],
        default="dense",
    )
    parser.add_argument("--score_method", choices=["attention", "key_norm", "random", "snapkv"], default="key_norm")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1, help="Useful for quick PG-19 single/few-sample runs.")
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--debug_selection", action="store_true")
    return parser


def load_texts(dataset_name: str, split: str, num_samples: int):
    if dataset_name == "wikitext":
        try:
            from datasets import load_dataset

            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except PermissionError:
            dataset = None
        if dataset is None:
            text = load_cached_wikitext_arrow(split)
            return [text]
        text = "\n\n".join(t for t in dataset["text"] if t.strip())
        return [text]
    from datasets import load_dataset

    dataset = load_dataset("pg19", split=split)
    texts = []
    for row in dataset.select(range(min(num_samples, len(dataset)))):
        texts.append(row.get("text", ""))
    return texts


def load_cached_wikitext_arrow(split: str) -> str:
    """Read a cached WikiText Arrow file without creating HF datasets locks.

    In the Codex sandbox, reading ``~/.cache`` is allowed but writing lock files
    there may be blocked.  This fallback keeps the evaluation reproducible when
    WikiText is already cached locally.
    """

    import pyarrow as pa

    candidates: List[Path] = list(
        (Path.home() / ".cache/huggingface/datasets/wikitext").glob(
            f"wikitext-2-raw-v1/*/*/wikitext-{split}.arrow"
        )
    )
    if not candidates:
        raise RuntimeError(
            "WikiText is not available through datasets and no cached Arrow file was found. "
            "Install/download it with normal HuggingFace permissions first."
        )
    path = candidates[-1]
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
    return "\n\n".join(t for t in table.column("text").to_pylist() if t and t.strip())


def evaluate_window(model, input_ids, prefill_start, prefill_end, eval_end, config, dense):
    device = input_ids.device
    prefill_ids = input_ids[:, prefill_start:prefill_end]
    eval_ids = input_ids[:, prefill_end:eval_end]
    if prefill_ids.numel() == 0 or eval_ids.numel() == 0:
        return [], None

    output_attentions = (not dense) and config.score_method in ("attention", "snapkv")
    original_attn_impl = getattr(model.config, "_attn_implementation", None)
    if output_attentions and original_attn_impl is not None:
        model.config._attn_implementation = "eager"
    with torch.no_grad():
        outputs = model(
            input_ids=prefill_ids,
            attention_mask=torch.ones_like(prefill_ids, device=device),
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        original_cache_len = cache_seq_length(past_key_values)
        if not dense:
            compression_start = time.perf_counter()
            past_key_values, stats = compress_past_key_values(
                past_key_values,
                config,
                getattr(outputs, "attentions", None),
            )
            stats["compression_overhead_sec"] = time.perf_counter() - compression_start
            stats["scoring_overhead_sec"] = 0.0
            if output_attentions:
                stats["scoring_note"] = (
                    "SnapKV/attention scoring used attention weights returned by the prefill pass; "
                    "its overhead is included in window prefill time."
                )
                if getattr(model.config, "_attn_implementation", None) == "eager":
                    model.config._attn_implementation = "sdpa"
                    stats["decode_attention_implementation"] = "sdpa"
            compressed_cache_len = cache_seq_length(past_key_values)
            assert compressed_cache_len <= prefill_ids.shape[1]
            if config.compression_ratio < 0.999 and compressed_cache_len == original_cache_len:
                stats["note"] = "KV cache was not shortened because mandatory/safe tokens met or exceeded the budget."
            elif config.compression_ratio < 0.999:
                assert compressed_cache_len < original_cache_len, "non-dense PPL path did not compress KV cache"
        else:
            stats = None
        next_token_logits = outputs.logits[:, -1, :]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nlls = []
    logical_seq_len = int(prefill_ids.shape[1])

    for idx in range(eval_ids.shape[1]):
        target = eval_ids[:, idx]
        nlls.append(float(loss_fct(next_token_logits, target).item()))
        position_ids = torch.full((1, 1), logical_seq_len, dtype=torch.long, device=device)
        cache_position = torch.arange(logical_seq_len, logical_seq_len + 1, dtype=torch.long, device=device)
        attention_mask = decode_attention_mask_for_cache(past_key_values, device)
        model_kwargs = {
            "input_ids": target.view(1, 1),
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "use_cache": True,
            "return_dict": True,
        }
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            out = model(**model_kwargs)
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        logical_seq_len += 1
    if output_attentions and original_attn_impl is not None:
        model.config._attn_implementation = original_attn_impl
    return nlls, stats


def evaluate_window_no_cache(model, input_ids, prefill_start, prefill_end, eval_end):
    """Score continuation tokens with ``use_cache=False``.

    For each target token, the model sees the full prefix up to that point and
    does not return or reuse ``past_key_values``.
    """

    eval_ids = input_ids[:, prefill_end:eval_end]
    if eval_ids.numel() == 0:
        return []

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nlls = []
    with torch.no_grad():
        for idx in range(eval_ids.shape[1]):
            prefix = input_ids[:, prefill_start : prefill_end + idx]
            target = eval_ids[:, idx]
            outputs = model(input_ids=prefix, use_cache=False)
            nlls.append(float(loss_fct(outputs.logits[:, -1, :], target).item()))
    return nlls


def main():
    args = build_arg_parser().parse_args()
    budget_mode = normalize_budget_mode(args.budget_mode)
    dense = budget_mode == "dense"
    no_cache = budget_mode == "no_cache"
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager" if args.score_method in ("attention", "snapkv") else None,
    )
    config = PyramidSinkKVConfig(
        compression_ratio=1.0 if dense or no_cache else args.compression_ratio,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        observation_window=args.observation_window,
        budget_mode=budget_mode,
        score_method=args.score_method,
        seed=args.seed,
        debug_selection=args.debug_selection,
    )

    texts = load_texts(args.dataset, args.split, args.num_samples)
    all_nlls = []
    compression_stats = []
    windows = 0
    for sample_idx, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        total = ids.shape[1]
        if total <= args.max_length:
            print(f"[ppl] sample {sample_idx} is shorter than max_length; skipping")
            continue
        start = 0
        while start + args.max_length < total and windows < args.max_windows:
            prefill_end = start + args.max_length
            eval_end = min(prefill_end + args.stride, total)
            if no_cache:
                nlls = evaluate_window_no_cache(model, ids, start, prefill_end, eval_end)
                stats = None
            else:
                nlls, stats = evaluate_window(model, ids, start, prefill_end, eval_end, config, dense)
            all_nlls.extend(nlls)
            if stats is not None:
                compression_stats.append(stats)
            windows += 1
            print(f"[ppl] window {windows}: scored {len(nlls)} tokens")
            start += args.stride

    if not all_nlls:
        raise RuntimeError("No tokens were evaluated. Try smaller --max_length or larger/longer dataset sample.")

    mean_nll = sum(all_nlls) / len(all_nlls)
    ppl = math.exp(mean_nll)
    payload = {
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset,
        "split": args.split,
        "budget_mode": budget_mode,
        "score_method": args.score_method,
        "observation_window": args.observation_window,
        "sink_size": args.sink_size,
        "recent_size": args.recent_size,
        "compression_ratio": 1.0 if dense or no_cache else args.compression_ratio,
        "max_length": args.max_length,
        "stride": args.stride,
        "num_tokens": len(all_nlls),
        "evaluated_tokens": len(all_nlls),
        "num_windows": windows,
        "negative_log_likelihood": mean_nll,
        "nll": mean_nll,
        "perplexity": ppl,
        "ppl": ppl,
        "score_method_fallback": any(item.get("score_method_fallback") for item in compression_stats),
        "fallback_score_method": "key_norm" if any(item.get("score_method_fallback") for item in compression_stats) else None,
        "compression": compression_stats[:1],
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
