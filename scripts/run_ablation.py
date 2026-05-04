import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    {"method": "no cache", "budget_mode": "no_cache", "score_method": "key_norm", "sink_size": 0, "recent_size": 0, "notes": "No KV cache; recomputes full prefix every token"},
    {"method": "dense", "budget_mode": "dense", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Dense baseline"},
    {"method": "random uniform", "budget_mode": "uniform", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Uniform random middle tokens"},
    {"method": "key_norm uniform", "budget_mode": "uniform", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Uniform SnapKV-style budget"},
    {"method": "random pyramid", "budget_mode": "pyramid", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Pyramid budget with random selection"},
    {"method": "key_norm pyramid", "budget_mode": "pyramid", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Main PyramidSinkKV variant"},
    {"method": "key_norm reversed", "budget_mode": "reversed", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Reversed-pyramid ablation"},
    {"method": "random spindle", "budget_mode": "spindle", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Spindle budget: middle layers keep more"},
    {"method": "key_norm spindle", "budget_mode": "spindle", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Spindle budget: middle layers keep more"},
    {"method": "random hourglass", "budget_mode": "hourglass", "score_method": "random", "sink_size": 4, "recent_size": 64, "notes": "Hourglass budget: edge layers keep more"},
    {"method": "key_norm hourglass", "budget_mode": "hourglass", "score_method": "key_norm", "sink_size": 4, "recent_size": 64, "notes": "Hourglass budget: edge layers keep more"},
    {"method": "pyramid without sink", "budget_mode": "pyramid", "score_method": "key_norm", "sink_size": 0, "recent_size": 64, "notes": "No-sink ablation"},
    {"method": "pyramid without recent", "budget_mode": "pyramid", "score_method": "key_norm", "sink_size": 4, "recent_size": 0, "notes": "No-recent-window ablation"},
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a small PyramidSinkKV ablation table.")
    parser.add_argument("--model_name_or_path", default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", default=("Efficient inference for language models is useful because " * 64))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_windows", type=int, default=2)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_seeds", default="0,1,2,3,4")
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


def mean_std(values):
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(values, precision=4, suffix=""):
    mean, std = mean_std(values)
    if mean is None:
        return "TODO"
    if len(values) == 1:
        return f"{mean:.{precision}f}{suffix}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}{suffix}"


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
        "--max_length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--compression_ratio",
        str(args.compression_ratio),
        "--sink_size",
        str(ablation["sink_size"]),
        "--recent_size",
        str(ablation["recent_size"]),
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
        str(args.compression_ratio),
        "--sink_size",
        str(ablation["sink_size"]),
        "--recent_size",
        str(ablation["recent_size"]),
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


def markdown_table(rows):
    headers = [
        "Method",
        "Budget mode",
        "Score method",
        "Sink size",
        "Recent size",
        "Compression ratio",
        "PPL",
        "TTFT",
        "TPOT",
        "Throughput",
        "KV memory",
        "Notes",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Method"]),
                    str(row["Budget mode"]),
                    str(row["Score method"]),
                    str(row["Sink size"]),
                    str(row["Recent size"]),
                    str(row["Compression ratio"]),
                    str(row["PPL"]),
                    str(row["TTFT"]),
                    str(row["TPOT"]),
                    str(row["Throughput"]),
                    str(row["KV memory"]),
                    str(row["Notes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    rows = []
    raw_records = []
    random_seeds = parse_seed_list(args.random_seeds)
    for idx, ablation in enumerate(ABLATIONS):
        name = ablation["method"].replace(" ", "_")
        dense_or_no_cache = ablation["budget_mode"] in ("dense", "no_cache")
        ratio = 1.0 if dense_or_no_cache else args.compression_ratio

        print(f"[ablation] {ablation['method']}")
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
        kv_values = [float(item.get("kv_cache_memory_mb", 0.0)) for item in gen_items]

        row = {
            "Method": ablation["method"],
            "Budget mode": ablation["budget_mode"],
            "Score method": ablation["score_method"],
            "Sink size": ablation["sink_size"],
            "Recent size": ablation["recent_size"],
            "Compression ratio": ratio,
            "PPL": "TODO",
            "TTFT": "TODO",
            "TPOT": "TODO",
            "Throughput": "TODO",
            "KV memory": "TODO",
            "Notes": ablation["notes"],
        }
        record = {
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
                "kv_memory_mean": mean_std(kv_values)[0],
                "kv_memory_std": mean_std(kv_values)[1],
                "ppl_seeds": seeds,
                "generation_repeats": max(1, args.generation_repeats),
            },
        }
        raw_records.append(record)

        row["PPL"] = format_mean_std(ppl_values, precision=4)
        row["TTFT"] = format_mean_std(ttft_values, precision=4)
        row["TPOT"] = format_mean_std(tpot_values, precision=4)
        row["Throughput"] = format_mean_std(throughput_values, precision=2)
        row["KV memory"] = format_mean_std(kv_values, precision=2, suffix=" MB")
        failed_ppl = [result for result in ppl_results if not result["ok"]]
        failed_gen = [result for result in gen_results if not result["ok"]]
        if ablation["score_method"] == "random":
            row["Notes"] += f" | PPL seeds={','.join(str(seed) for seed in seeds)}"
        if max(1, args.generation_repeats) > 1:
            row["Notes"] += f" | generation repeats={max(1, args.generation_repeats)}"
        if failed_ppl:
            row["Notes"] += " | PPL failed: " + failed_ppl[0]["error"].splitlines()[-1][:120]
        if failed_gen:
            row["Notes"] += " | Generation failed: " + failed_gen[0]["error"].splitlines()[-1][:120]
        rows.append(row)

    (results_dir / "ablation_results.json").write_text(json.dumps(raw_records, indent=2, ensure_ascii=False), encoding="utf-8")
    with (results_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    print(f"[ablation] wrote {results_dir / 'ablation_table.md'}")


if __name__ == "__main__":
    main()
