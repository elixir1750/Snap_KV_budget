import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


MAIN_ABLATIONS = [
    {"method": "dense", "budget_mode": "dense", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Dense KV-cache baseline"},
    {"method": "no_cache", "budget_mode": "no_cache", "score_method": "key_norm", "sink_size": 0, "recent_size": 0, "notes": "Optional no-cache baseline; recomputes full prefix every token"},
    {"method": "uniform random", "budget_mode": "uniform", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Uniform budget with random middle-token selection"},
    {"method": "uniform snapkv", "budget_mode": "uniform", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Uniform budget with SnapKV attention-based selection"},
    {"method": "pyramid random", "budget_mode": "pyramid", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Pyramid budget with random middle-token selection"},
    {"method": "pyramid snapkv", "budget_mode": "pyramid", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Pyramid budget with SnapKV attention-based selection"},
    {"method": "spindle random", "budget_mode": "spindle", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Spindle budget with random middle-token selection"},
    {"method": "spindle snapkv", "budget_mode": "spindle", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Spindle budget with SnapKV attention-based selection"},
    {"method": "reversed snapkv", "budget_mode": "reversed", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Reversed budget with SnapKV attention-based selection"},
    {"method": "hourglass snapkv", "budget_mode": "hourglass", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Hourglass budget with SnapKV attention-based selection"},
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run grouped PyramidSinkKV/SnapKV ablation tables.")
    parser.add_argument("--model_name_or_path", default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", default=("Efficient inference for language models is useful because " * 64))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dataset", choices=["wikitext", "pg19", "redpajama"], default="wikitext")
    parser.add_argument("--split", default="test")
    parser.add_argument("--redpajama_file", default="book_sample.jsonl")
    parser.add_argument("--redpajama_source", choices=["sample_file", "hub"], default="sample_file")
    parser.add_argument("--redpajama_hub_dataset", default="togethercomputer/RedPajama-Data-1T")
    parser.add_argument("--redpajama_hub_config", default="wikipedia")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--observation_window", type=int, default=32)
    parser.add_argument("--snapkv_pooling_kernel", type=int, default=1)
    parser.add_argument("--snapkv_head_aggregation", choices=["mean", "max", "per_head"], default="mean")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_windows", type=int, default=2)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_seeds", default="0,1,2,3,4")
    parser.add_argument("--score_methods", default="random,snapkv")
    parser.add_argument("--budget_modes", default="uniform,pyramid,reversed,spindle,hourglass")
    parser.add_argument("--generation_repeats", type=int, default=3)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def run_command(cmd, output_path):
    if output_path.exists():
        output_path.unlink()
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}",
        }
    if not output_path.exists():
        return {"ok": False, "error": f"Expected output file was not created: {output_path}"}
    return {"ok": True, "data": json.loads(output_path.read_text(encoding="utf-8"))}


def extract_benchmark(path_data):
    results = path_data.get("results", [])
    return results[0] if results else {}


def parse_seed_list(seed_text):
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def parse_name_list(text):
    return {item.strip() for item in text.split(",") if item.strip()}


def ablation_ratio(args, ablation):
    if ablation["budget_mode"] in ("dense", "no_cache"):
        return 1.0
    return float(ablation.get("compression_ratio", args.compression_ratio))


def build_ratio_sweep(score_methods, budget_modes):
    items = [
        {"method": "dense", "budget_mode": "dense", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Dense KV-cache baseline"}
    ]
    for ratio in (0.75, 0.5, 0.25):
        for mode, score in (("uniform", "random"), ("uniform", "snapkv"), ("spindle", "random"), ("spindle", "snapkv")):
            if mode in budget_modes and score in score_methods:
                items.append(
                    {
                        "method": f"{mode} {score} ratio {ratio}",
                        "budget_mode": mode,
                        "score_method": score,
                        "sink_size": 4,
                        "recent_size": 64,
                        "compression_ratio": ratio,
                        "notes": f"Compression-ratio sweep at keep ratio {ratio}",
                    }
                )
    return items


def build_sink_recent_ablation(score_methods, budget_modes):
    if "snapkv" not in score_methods or "spindle" not in budget_modes:
        return []
    return [
        {"method": "spindle snapkv sink recent", "budget_mode": "spindle", "score_method": "snapkv", "sink_size": 4, "recent_size": 64, "notes": "Default sink + recent policy"},
        {"method": "spindle snapkv no sink", "budget_mode": "spindle", "score_method": "snapkv", "sink_size": 0, "recent_size": 64, "notes": "No sink preservation"},
        {"method": "spindle snapkv no recent", "budget_mode": "spindle", "score_method": "snapkv", "sink_size": 4, "recent_size": 0, "notes": "No recent-window preservation"},
        {"method": "spindle snapkv no sink no recent", "budget_mode": "spindle", "score_method": "snapkv", "sink_size": 0, "recent_size": 0, "notes": "No sink and no recent preservation"},
    ]


def mean_std(values):
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(values, precision=4):
    mean, std = mean_std(values)
    if mean is None:
        return "TODO"
    if len(values) == 1:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def run_ppl_once(args, ablation, raw_dir, idx, name, seed):
    ppl_json = raw_dir / f"{idx:02d}_{name}_seed{seed}_ppl.json"
    if args.skip_existing and ppl_json.exists():
        return {"ok": True, "data": json.loads(ppl_json.read_text(encoding="utf-8")), "seed": seed}
    ppl_cmd = [
        sys.executable,
        "-m",
        "scripts.eval_ppl",
        "--model_name_or_path",
        args.model_name_or_path,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--redpajama_file",
        args.redpajama_file,
        "--redpajama_source",
        args.redpajama_source,
        "--redpajama_hub_dataset",
        args.redpajama_hub_dataset,
        "--redpajama_hub_config",
        args.redpajama_hub_config,
        "--max_length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--compression_ratio",
        str(ablation_ratio(args, ablation)),
        "--sink_size",
        str(ablation["sink_size"]),
        "--recent_size",
        str(ablation["recent_size"]),
        "--observation_window",
        str(args.observation_window),
        "--snapkv_pooling_kernel",
        str(args.snapkv_pooling_kernel),
        "--snapkv_head_aggregation",
        args.snapkv_head_aggregation,
        "--budget_mode",
        ablation["budget_mode"],
        "--score_method",
        ablation["score_method"],
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--num_samples",
        str(args.num_samples),
        "--max_windows",
        str(args.max_windows),
        "--output_json",
        str(ppl_json),
    ]
    result = run_command(ppl_cmd, ppl_json)
    result["seed"] = seed
    return result


def run_generation_once(args, ablation, raw_dir, idx, name, repeat_idx):
    gen_json = raw_dir / f"{idx:02d}_{name}_generation_repeat{repeat_idx}.json"
    if args.skip_existing and gen_json.exists():
        return {"ok": True, "data": json.loads(gen_json.read_text(encoding="utf-8")), "repeat": repeat_idx}
    gen_cmd = [
        sys.executable,
        "-m",
        "scripts.benchmark_generation",
        "--model_name_or_path",
        args.model_name_or_path,
        "--prompt",
        args.prompt,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--compression_ratio",
        str(ablation_ratio(args, ablation)),
        "--sink_size",
        str(ablation["sink_size"]),
        "--recent_size",
        str(ablation["recent_size"]),
        "--observation_window",
        str(args.observation_window),
        "--snapkv_pooling_kernel",
        str(args.snapkv_pooling_kernel),
        "--snapkv_head_aggregation",
        args.snapkv_head_aggregation,
        "--budget_mode",
        ablation["budget_mode"],
        "--score_method",
        ablation["score_method"],
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--warmup_runs",
        "0",
        "--output_json",
        str(gen_json),
    ]
    result = run_command(gen_cmd, gen_json)
    result["repeat"] = repeat_idx
    return result


TABLE_FIELDS = [
    "method",
    "budget_mode",
    "score_method",
    "sink_size",
    "recent_size",
    "compression_ratio",
    "ppl",
    "ttft",
    "tpot",
    "throughput",
    "total_time",
    "kv_cache_memory_mb",
    "achieved_compression_ratio",
    "scoring_overhead_sec",
    "snapkv_fallback_status",
    "notes",
]


def markdown_table(rows):
    lines = ["| " + " | ".join(TABLE_FIELDS) + " |", "| " + " | ".join(["---"] * len(TABLE_FIELDS)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in TABLE_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def run_group(args, results_dir, group_name, output_prefix, ablations):
    raw_dir = results_dir / "raw" / group_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_records = []
    random_seeds = parse_seed_list(args.random_seeds)
    for idx, ablation in enumerate(ablations):
        name = ablation["method"].replace(" ", "_").replace(".", "_")
        ratio = ablation_ratio(args, ablation)

        print(f"[ablation:{group_name}] {ablation['method']}")
        seeds = random_seeds if ablation["score_method"] == "random" else [args.seed]
        ppl_results = [run_ppl_once(args, ablation, raw_dir, idx, name, seed) for seed in seeds]
        gen_results = [
            run_generation_once(args, ablation, raw_dir, idx, name, repeat_idx)
            for repeat_idx in range(max(1, args.generation_repeats))
        ]

        ppl_values = [float(result["data"]["perplexity"]) for result in ppl_results if result["ok"]]
        gen_items = [extract_benchmark(result["data"]) for result in gen_results if result["ok"]]
        ttft_values = [float(item.get("ttft", 0.0)) for item in gen_items]
        tpot_values = [float(item.get("tpot", 0.0)) for item in gen_items]
        throughput_values = [float(item.get("throughput", 0.0)) for item in gen_items]
        total_time_values = [float(item.get("total_time", 0.0)) for item in gen_items]
        kv_values = [float(item.get("kv_cache_memory_mb", 0.0)) for item in gen_items]
        achieved_values = [
            float(item["achieved_compression_ratio"])
            for item in gen_items
            if item.get("achieved_compression_ratio") is not None
        ]
        scoring_values = [float(item.get("scoring_overhead_sec", 0.0)) for item in gen_items]
        fallback_happened = any(
            bool(result.get("data", {}).get("score_method_fallback"))
            for result in ppl_results
            if result["ok"]
        ) or any(bool(item.get("score_method_fallback")) for item in gen_items)
        fallback_status = "not_applicable"
        if ablation["score_method"] == "snapkv":
            fallback_status = "fallback_to_key_norm" if fallback_happened else "true_snapkv"

        notes = ablation["notes"]
        if ablation["score_method"] == "random":
            notes += f" | PPL seeds={','.join(str(seed) for seed in seeds)}"
        if fallback_happened and ablation["score_method"] == "snapkv":
            notes += " | SnapKV attention unavailable; fell back to key_norm, so this is not a true SnapKV result"
        if max(1, args.generation_repeats) > 1:
            notes += f" | generation repeats={max(1, args.generation_repeats)}"
        failed_ppl = [result for result in ppl_results if not result["ok"]]
        failed_gen = [result for result in gen_results if not result["ok"]]
        if failed_ppl:
            notes += " | PPL failed: " + failed_ppl[0]["error"].splitlines()[-1][:120]
        if failed_gen:
            notes += " | Generation failed: " + failed_gen[0]["error"].splitlines()[-1][:120]

        row = {
            "method": ablation["method"],
            "budget_mode": ablation["budget_mode"],
            "score_method": "none" if ablation["budget_mode"] in ("dense", "no_cache") else ablation["score_method"],
            "sink_size": ablation["sink_size"],
            "recent_size": ablation["recent_size"],
            "compression_ratio": ratio,
            "ppl": format_mean_std(ppl_values, precision=4),
            "ttft": format_mean_std(ttft_values, precision=4),
            "tpot": format_mean_std(tpot_values, precision=4),
            "throughput": format_mean_std(throughput_values, precision=2),
            "total_time": format_mean_std(total_time_values, precision=4),
            "kv_cache_memory_mb": format_mean_std(kv_values, precision=2),
            "achieved_compression_ratio": format_mean_std(achieved_values, precision=4),
            "scoring_overhead_sec": format_mean_std(scoring_values, precision=4),
            "snapkv_fallback_status": fallback_status,
            "notes": notes,
        }
        rows.append(row)
        raw_records.append(
            {
                "ablation": ablation,
                "ppl_runs": ppl_results,
                "generation_runs": gen_results,
                "summary": {
                    "ppl_mean": mean_std(ppl_values)[0],
                    "ppl_std": mean_std(ppl_values)[1],
                    "ttft_mean": mean_std(ttft_values)[0],
                    "ttft_std": mean_std(ttft_values)[1],
                    "tpot_mean": mean_std(tpot_values)[0],
                    "tpot_std": mean_std(tpot_values)[1],
                    "throughput_mean": mean_std(throughput_values)[0],
                    "throughput_std": mean_std(throughput_values)[1],
                    "total_time_mean": mean_std(total_time_values)[0],
                    "total_time_std": mean_std(total_time_values)[1],
                    "kv_cache_memory_mb_mean": mean_std(kv_values)[0],
                    "kv_cache_memory_mb_std": mean_std(kv_values)[1],
                    "achieved_compression_ratio_mean": mean_std(achieved_values)[0],
                    "achieved_compression_ratio_std": mean_std(achieved_values)[1],
                    "scoring_overhead_sec_mean": mean_std(scoring_values)[0],
                    "scoring_overhead_sec_std": mean_std(scoring_values)[1],
                    "snapkv_fallback_status": fallback_status,
                    "ppl_seeds": seeds,
                    "generation_repeats": max(1, args.generation_repeats),
                },
            }
        )

    json_path = results_dir / f"{output_prefix}.json"
    csv_path = results_dir / f"{output_prefix}.csv"
    md_path = results_dir / f"{output_prefix}.md"
    json_path.write_text(json.dumps(raw_records, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    print(f"[ablation:{group_name}] wrote {md_path}")


def main():
    args = build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    score_methods = parse_name_list(args.score_methods)
    budget_modes = parse_name_list(args.budget_modes)
    main_ablations = [
        item
        for item in MAIN_ABLATIONS
        if item["budget_mode"] in ("dense", "no_cache")
        or (item["budget_mode"] in budget_modes and item["score_method"] in score_methods)
    ]
    groups = [
        ("main", "group_main_ablation", main_ablations),
        ("ratio_sweep", "group_ratio_sweep", build_ratio_sweep(score_methods, budget_modes)),
        ("sink_recent", "group_sink_recent_ablation", build_sink_recent_ablation(score_methods, budget_modes)),
    ]
    for group_name, output_prefix, ablations in groups:
        if not ablations:
            print(f"[ablation:{group_name}] skipped; no experiments matched requested filters")
            continue
        run_group(args, results_dir, group_name, output_prefix, ablations)


if __name__ == "__main__":
    main()
