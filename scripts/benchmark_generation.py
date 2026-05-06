import argparse
import json
from pathlib import Path

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
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--observation_window", type=int, default=32)
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
                budget_mode="pyramid",
                score_method="key_norm",
                seed=args.seed,
            ),
        ),
    ]


def main():
    args = build_arg_parser().parse_args()
    requested_attn = args.score_method in ("attention", "snapkv")
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager" if requested_attn else None,
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
        generate(model, tokenizer, args.prompt, min(args.max_new_tokens, 8), methods[0][1], device)

    results = []
    for method_name, config in methods:
        print(f"[benchmark] running {method_name}")
        if config.budget_mode == "no_cache":
            item = generate_no_cache(model, tokenizer, args.prompt, args.max_new_tokens, device)
        else:
            item = generate(model, tokenizer, args.prompt, args.max_new_tokens, config, device)
        item["method"] = method_name
        results.append(item)
        print(
            f"  TTFT={item['ttft']:.4f}s TPOT={item['tpot']:.4f}s "
            f"throughput={item['throughput']:.2f} tok/s KV={item['kv_cache_memory_mb']:.2f} MB "
            f"scoring_overhead={item.get('compression', {}).get('scoring_overhead_sec', 0.0):.4f}s "
            f"compression_overhead={item.get('compression', {}).get('compression_overhead_sec', 0.0):.4f}s"
        )

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "max_new_tokens": args.max_new_tokens,
        "observation_window": args.observation_window,
        "prompt": args.prompt,
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
