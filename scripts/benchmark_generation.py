import argparse
import json
from pathlib import Path

import torch

from pyramidsinkkv import (
    PyramidSinkKVConfig,
    generate,
    generate_no_cache,
    load_model_and_tokenizer,
    normalize_budget_mode,
)


DEFAULT_PROMPT = (
    "Training-free efficient inference keeps the language model weights fixed while changing only "
    "the key value cache used during autoregressive decoding. "
) * 16


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dense and PyramidSinkKV generation for Pythia/GPTNeoX.")
    parser.add_argument("--model_name_or_path", default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt_file", default=None, help="Optional text file used as the generation prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument(
        "--snapkv_pooling_kernel",
        type=int,
        default=1,
        help="Length-preserving 1D average-pooling kernel for SnapKV scores; 1 disables pooling.",
    )
    parser.add_argument(
        "--snapkv_head_aggregation",
        choices=["mean", "max", "per_head"],
        default="mean",
        help="How to use per-head SnapKV scores: mean/max share one index set; per_head gathers distinct indices per head.",
    )
    parser.add_argument(
        "--budget_mode",
        choices=["no_cache", "dense", "uniform", "pyramid", "reversed", "spindle", "hourglass"],
        default=None,
    )
    parser.add_argument("--score_method", choices=["attention", "key_norm", "random", "snapkv"], default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--debug_selection", action="store_true")
    return parser


def load_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def add_timing_memory_metrics(item, device):
    tpot = float(item.get("tpot", 0.0) or 0.0)
    total_time = float(item.get("total_time", 0.0) or 0.0)
    generated_tokens = int(item.get("generated_tokens", 0) or 0)
    prompt_tokens = int(item.get("prompt_tokens", 0) or 0)
    item["decode_tokens_per_sec"] = (1.0 / tpot) if tpot > 0 else None
    item["total_tokens_per_sec"] = ((prompt_tokens + generated_tokens) / total_time) if total_time > 0 else None
    item["estimated_kv_cache_memory_mb"] = item.get("kv_cache_memory_mb")
    if device.type == "cuda":
        item["peak_cuda_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        item["peak_cuda_memory_allocated_mb"] = None
    return item


def default_methods(args):
    return [
        ("no_cache", PyramidSinkKVConfig(compression_ratio=1.0, budget_mode="no_cache")),
        ("dense", PyramidSinkKVConfig(compression_ratio=1.0, budget_mode="dense")),
        (
            "uniform_key_norm",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="uniform",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "uniform_random",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="uniform",
                score_method="random",
                seed=args.seed,
            ),
        ),
        (
            "pyramid_key_norm",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="pyramid",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "pyramid_random",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="pyramid",
                score_method="random",
                seed=args.seed,
            ),
        ),
        (
            "reversed_key_norm",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="reversed",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "spindle_key_norm",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="spindle",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "spindle_random",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="spindle",
                score_method="random",
                seed=args.seed,
            ),
        ),
        (
            "hourglass_key_norm",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="hourglass",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "hourglass_random",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="hourglass",
                score_method="random",
                seed=args.seed,
            ),
        ),
        (
            "pyramid_key_norm_no_sink",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=0,
                recent_size=args.recent_size,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="pyramid",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
        (
            "pyramid_key_norm_no_recent",
            PyramidSinkKVConfig(
                compression_ratio=args.compression_ratio,
                sink_size=args.sink_size,
                recent_size=0,
                observation_window=args.observation_window,
                snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                snapkv_head_aggregation=args.snapkv_head_aggregation,
                budget_mode="pyramid",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
    ]


def main():
    args = build_arg_parser().parse_args()
    prompt = load_prompt(args)
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=None,
    )

    if args.budget_mode is not None or args.score_method is not None:
        mode = normalize_budget_mode(args.budget_mode or "pyramid")
        score_method = args.score_method or "key_norm"
        methods = [
            (
                f"{mode}_{score_method}",
                PyramidSinkKVConfig(
                    compression_ratio=1.0 if mode in ("dense", "no_cache") else args.compression_ratio,
                    sink_size=args.sink_size,
                    recent_size=args.recent_size,
                    observation_window=args.observation_window,
                    snapkv_pooling_kernel=args.snapkv_pooling_kernel,
                    snapkv_head_aggregation=args.snapkv_head_aggregation,
                    budget_mode=mode,
                    score_method=score_method,
                    seed=args.seed,
                    debug_selection=args.debug_selection,
                ),
            )
        ]
    else:
        methods = default_methods(args)

    for _ in range(max(0, args.warmup_runs)):
        generate(model, tokenizer, prompt, min(args.max_new_tokens, 8), methods[0][1], device)

    results = []
    for method_name, config in methods:
        print(f"[benchmark] running {method_name}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        if config.budget_mode == "no_cache":
            item = generate_no_cache(model, tokenizer, prompt, args.max_new_tokens, device)
        else:
            item = generate(model, tokenizer, prompt, args.max_new_tokens, config, device)
        item["method"] = method_name
        add_timing_memory_metrics(item, device)
        results.append(item)
        print(
            f"  TTFT={item['ttft']:.4f}s TPOT={item['tpot']:.4f}s "
            f"decode={item['decode_tokens_per_sec'] or 0.0:.2f} tok/s "
            f"total={item['total_tokens_per_sec'] or 0.0:.2f} tok/s "
            f"KV={item['estimated_kv_cache_memory_mb']:.2f} MB "
            f"scoring_overhead={item.get('compression', {}).get('scoring_overhead_sec', 0.0):.4f}s "
            f"compression_overhead={item.get('compression', {}).get('compression_overhead_sec', 0.0):.4f}s"
        )

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "max_new_tokens": args.max_new_tokens,
        "observation_window": args.observation_window,
        "snapkv_pooling_kernel": args.snapkv_pooling_kernel,
        "snapkv_head_aggregation": args.snapkv_head_aggregation,
        "prompt": prompt,
        "prompt_file": args.prompt_file,
        "results": results,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
