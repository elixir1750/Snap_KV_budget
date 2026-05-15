#!/usr/bin/env bash
set -u

# RedPajama-arXiv length ablation for spindle + SnapKV variants.
#
# Default method combinations per context length:
# - dense full-cache baseline
# - spindle + random, seeds 0-4
# - spindle + snapkv mean pool=1
# - spindle + snapkv mean pool=7
# - spindle + snapkv max pool=7
# - spindle + snapkv per_head pool=7
#
# The pooled variants use SNAPKV_POOLING_KERNEL=7 by default, which is closer
# to the SnapKV paper setting.  To reproduce the earlier pool=3 table exactly:
#
#   SNAPKV_POOLING_KERNEL=3 bash scripts/run_redpajama_arxiv_length_ablation.sh
#
# Useful overrides:
#   DEVICE=cpu DTYPE=float32 NUM_SAMPLES=2 MAX_WINDOWS=1 RUN_GENERATION_BENCHMARKS=0 bash scripts/run_redpajama_arxiv_length_ablation.sh
#   LENGTHS="512 1024" MAX_WINDOWS=1 bash scripts/run_redpajama_arxiv_length_ablation.sh
#   RANDOM_SEEDS="0" NUM_SAMPLES=2 bash scripts/run_redpajama_arxiv_length_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pyramidsinkkv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-EleutherAI/pythia-70m}"
RESULTS_DIR="${RESULTS_DIR:-results/redpajama_arxiv_spindle_length_ablation_ratio025}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-arxiv_length_ablation}"

DATASET="${DATASET:-redpajama}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-hub}"
REDPAJAMA_HUB_CONFIG="${REDPAJAMA_HUB_CONFIG:-arxiv}"
SPLIT="${SPLIT:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
LENGTHS="${LENGTHS:-512 1024 1536 2048}"
STRIDE="${STRIDE:-256}"
MAX_WINDOWS="${MAX_WINDOWS:-8}"

COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.25}"
SINK_SIZE="${SINK_SIZE:-4}"
RECENT_SIZE="${RECENT_SIZE:-64}"
OBSERVATION_WINDOW="${OBSERVATION_WINDOW:-32}"
SNAPKV_POOLING_KERNEL="${SNAPKV_POOLING_KERNEL:-7}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2 3 4}"
RUN_GENERATION_BENCHMARKS="${RUN_GENERATION_BENCHMARKS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
PLOT_METRICS="${PLOT_METRICS:-ppl decode_tokens_per_sec estimated_kv_cache_memory_mb}"

mkdir -p "${RESULTS_DIR}"

echo "[arxiv-length] results_dir=${RESULTS_DIR}"
echo "[arxiv-length] dataset=${DATASET} redpajama_hub_config=${REDPAJAMA_HUB_CONFIG}"
echo "[arxiv-length] lengths=${LENGTHS}"
echo "[arxiv-length] compression_ratio=${COMPRESSION_RATIO}"
echo "[arxiv-length] observation_window=${OBSERVATION_WINDOW}"
echo "[arxiv-length] pooled_snapkv_kernel=${SNAPKV_POOLING_KERNEL}"
echo "[arxiv-length] random_seeds=${RANDOM_SEEDS}"
echo "[arxiv-length] device=${DEVICE} dtype=${DTYPE}"
echo "[arxiv-length] run_generation_benchmarks=${RUN_GENERATION_BENCHMARKS} max_new_tokens=${MAX_NEW_TOKENS}"
echo "[arxiv-length] plot_metrics=${PLOT_METRICS}"

if [[ "${RUN_GENERATION_BENCHMARKS}" != "0" ]]; then
  prompt_dir="${RESULTS_DIR}/prompts"
  mkdir -p "${prompt_dir}"
  "${PYTHON_BIN}" - "${MODEL_NAME_OR_PATH}" "${prompt_dir}" "${LENGTHS}" "${DATASET}" "${SPLIT}" "${NUM_SAMPLES}" "${REDPAJAMA_SOURCE}" "${REDPAJAMA_HUB_CONFIG}" <<'PY'
import sys
from pathlib import Path

from transformers import AutoTokenizer

from scripts.eval_ppl import load_texts

model_name = sys.argv[1]
prompt_dir = Path(sys.argv[2])
lengths = [int(item) for item in sys.argv[3].split()]
dataset_name = sys.argv[4]
split = sys.argv[5]
num_samples = int(sys.argv[6])
redpajama_source = sys.argv[7]
redpajama_hub_config = sys.argv[8]

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = load_texts(
    dataset_name,
    split,
    num_samples,
    redpajama_source=redpajama_source,
    redpajama_hub_config=redpajama_hub_config,
)[0]
ids = tokenizer(text, return_tensors="pt").input_ids[0]
needed = max(lengths)
if int(ids.numel()) < needed:
    raise RuntimeError(f"{dataset_name} prompt has {int(ids.numel())} tokens, need at least {needed}.")
for length in lengths:
    prompt = tokenizer.decode(ids[:length], skip_special_tokens=False)
    out = prompt_dir / f"prompt_len{length}.txt"
    out.write_text(prompt, encoding="utf-8")
    print(f"[prompt] wrote {out} from {length} tokens")
PY
fi

run_eval() {
  local label="$1"
  local max_length="$2"
  local budget_mode="$3"
  local score_method="$4"
  local seed="$5"
  local pooling_kernel="$6"
  local head_aggregation="$7"
  local output_json="$8"
  local failed_marker="${output_json%.json}.failed.txt"

  if [[ -f "${output_json}" ]]; then
    echo "[skip] ${label}: ${output_json}"
    return 0
  fi

  echo "[run] ${label}: len=${max_length} budget=${budget_mode} score=${score_method} seed=${seed} aggregation=${head_aggregation} pool=${pooling_kernel}"
  "${PYTHON_BIN}" -m scripts.eval_ppl \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset "${DATASET}" \
    --redpajama_source "${REDPAJAMA_SOURCE}" \
    --redpajama_hub_config "${REDPAJAMA_HUB_CONFIG}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_length "${max_length}" \
    --stride "${STRIDE}" \
    --max_windows "${MAX_WINDOWS}" \
    --compression_ratio "${COMPRESSION_RATIO}" \
    --sink_size "${SINK_SIZE}" \
    --recent_size "${RECENT_SIZE}" \
    --observation_window "${OBSERVATION_WINDOW}" \
    --snapkv_pooling_kernel "${pooling_kernel}" \
    --snapkv_head_aggregation "${head_aggregation}" \
    --budget_mode "${budget_mode}" \
    --score_method "${score_method}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --output_json "${output_json}"

  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "${label} failed with exit code ${status}" > "${failed_marker}"
    echo "[failed] ${label}; wrote ${failed_marker}"
  else
    rm -f "${failed_marker}"
  fi
  return 0
}

run_benchmark() {
  local label="$1"
  local max_length="$2"
  local budget_mode="$3"
  local score_method="$4"
  local seed="$5"
  local pooling_kernel="$6"
  local head_aggregation="$7"
  local output_json="$8"
  local prompt_file="${RESULTS_DIR}/prompts/prompt_len${max_length}.txt"
  local failed_marker="${output_json%.json}.failed.txt"

  if [[ "${RUN_GENERATION_BENCHMARKS}" == "0" ]]; then
    return 0
  fi
  if [[ -f "${output_json}" ]]; then
    echo "[skip] ${label}: ${output_json}"
    return 0
  fi

  echo "[bench] ${label}: len=${max_length} budget=${budget_mode} score=${score_method} seed=${seed} aggregation=${head_aggregation} pool=${pooling_kernel}"
  "${PYTHON_BIN}" -m scripts.benchmark_generation \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --prompt_file "${prompt_file}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --compression_ratio "${COMPRESSION_RATIO}" \
    --sink_size "${SINK_SIZE}" \
    --recent_size "${RECENT_SIZE}" \
    --observation_window "${OBSERVATION_WINDOW}" \
    --snapkv_pooling_kernel "${pooling_kernel}" \
    --snapkv_head_aggregation "${head_aggregation}" \
    --budget_mode "${budget_mode}" \
    --score_method "${score_method}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --warmup_runs "${WARMUP_RUNS}" \
    --output_json "${output_json}"

  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "${label} generation benchmark failed with exit code ${status}" > "${failed_marker}"
    echo "[failed] ${label}; wrote ${failed_marker}"
  else
    rm -f "${failed_marker}"
  fi
  return 0
}

for max_length in ${LENGTHS}; do
  length_dir="${RESULTS_DIR}/len${max_length}"
  mkdir -p "${length_dir}"

  run_eval \
    "dense baseline" \
    "${max_length}" \
    "dense" \
    "key_norm" \
    "0" \
    "1" \
    "mean" \
    "${length_dir}/dense_full_cache_seed0_ppl.json"
  run_benchmark \
    "dense baseline" \
    "${max_length}" \
    "dense" \
    "key_norm" \
    "0" \
    "1" \
    "mean" \
    "${length_dir}/dense_full_cache_seed0_generation.json"

  for seed in ${RANDOM_SEEDS}; do
    run_eval \
      "spindle random seed ${seed}" \
      "${max_length}" \
      "spindle" \
      "random" \
      "${seed}" \
      "1" \
      "mean" \
      "${length_dir}/spindle_random_seed${seed}_ppl.json"
    run_benchmark \
      "spindle random seed ${seed}" \
      "${max_length}" \
      "spindle" \
      "random" \
      "${seed}" \
      "1" \
      "mean" \
      "${length_dir}/spindle_random_seed${seed}_generation.json"
  done

  run_eval \
    "spindle snapkv mean pool=1" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "1" \
    "mean" \
    "${length_dir}/spindle_snapkv_mean_pool1_seed0_ppl.json"
  run_benchmark \
    "spindle snapkv mean pool=1" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "1" \
    "mean" \
    "${length_dir}/spindle_snapkv_mean_pool1_seed0_generation.json"

  run_eval \
    "spindle snapkv mean pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "mean" \
    "${length_dir}/spindle_snapkv_mean_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"
  run_benchmark \
    "spindle snapkv mean pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "mean" \
    "${length_dir}/spindle_snapkv_mean_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"

  run_eval \
    "spindle snapkv max pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "max" \
    "${length_dir}/spindle_snapkv_max_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"
  run_benchmark \
    "spindle snapkv max pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "max" \
    "${length_dir}/spindle_snapkv_max_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"

  run_eval \
    "spindle snapkv per_head pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "per_head" \
    "${length_dir}/spindle_snapkv_per_head_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"
  run_benchmark \
    "spindle snapkv per_head pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "per_head" \
    "${length_dir}/spindle_snapkv_per_head_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${LENGTHS}" "${COMPRESSION_RATIO}" "${RANDOM_SEEDS}" "${SNAPKV_POOLING_KERNEL}" "${SUMMARY_PREFIX}" "${PLOT_METRICS}" <<'PY'
import csv
import html
import json
import math
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
lengths = [int(item) for item in sys.argv[2].split()]
compression_ratio = sys.argv[3]
random_seeds = sys.argv[4].split()
pooled_kernel = sys.argv[5]
summary_prefix = sys.argv[6]
plot_metrics = sys.argv[7].split()


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def failed_note(path):
    marker = Path(str(path).replace(".json", ".failed.txt"))
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    if not path.exists():
        return "missing output json"
    return ""


def finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean_std(values, precision=4):
    values = [value for value in values if value is not None]
    if not values:
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.{precision}f}"
    return f"{st.mean(values):.{precision}f} +/- {st.pstdev(values):.{precision}f}"


def parse_summary_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("NA", "FAILED", "nan", "None"):
        return None
    text = text.split("+/-", 1)[0].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def achieved_ratio(data, default="NA"):
    if not data or data.get("_error"):
        return default
    if data.get("budget_mode") == "dense":
        return "1.0000"
    compression = data.get("compression") or []
    if compression and compression[0].get("achieved_compression_ratio") is not None:
        return f"{float(compression[0]['achieved_compression_ratio']):.4f}"
    return default


def snapkv_status(data):
    if not data or data.get("_error"):
        return "unknown"
    if data.get("score_method") != "snapkv":
        return "not_applicable"
    return "fallback_to_key_norm" if data.get("score_method_fallback") else "true_snapkv"


def generation_item(path):
    data = read_json(path)
    if not data or data.get("_error"):
        return None
    results = data.get("results") or []
    return results[0] if results else None


def gen_metrics(paths):
    items = [generation_item(path) for path in paths]
    items = [item for item in items if item]
    return {
        "ttft": mean_std([finite_float(item.get("ttft")) for item in items]),
        "tpot": mean_std([finite_float(item.get("tpot")) for item in items], precision=6),
        "decode_tokens_per_sec": mean_std([finite_float(item.get("decode_tokens_per_sec")) for item in items]),
        "total_tokens_per_sec": mean_std([finite_float(item.get("total_tokens_per_sec")) for item in items]),
        "peak_cuda_memory_allocated_mb": mean_std([finite_float(item.get("peak_cuda_memory_allocated_mb")) for item in items]),
        "estimated_kv_cache_memory_mb": mean_std([finite_float(item.get("estimated_kv_cache_memory_mb")) for item in items]),
    }


def add_row(
    max_length,
    method,
    budget,
    selection,
    pooling,
    ratio,
    runs,
    ppl,
    achieved,
    status,
    notes,
    gen_paths,
):
    row = {
        "max_length": max_length,
        "method": method,
        "budget": budget,
        "selection": selection,
        "pooling": pooling,
        "ratio": ratio,
        "runs": runs,
        "ppl": ppl,
        "achieved_compression_ratio": achieved,
        "snapkv_fallback_status": status,
        "notes": notes,
    }
    row.update(gen_metrics(gen_paths))
    rows.append(row)


rows = []

for max_length in lengths:
    length_dir = root / f"len{max_length}"

    dense_path = length_dir / "dense_full_cache_seed0_ppl.json"
    dense = read_json(dense_path)
    if dense and not dense.get("_error"):
        dense_ppl = f"{float(dense['ppl']):.4f}"
        dense_notes = "baseline"
        dense_runs = 1
    else:
        dense_ppl = "FAILED"
        dense_notes = failed_note(dense_path) or (dense or {}).get("_error", "")
        dense_runs = 0
    add_row(
        max_length,
        "Dense",
        "full",
        "none",
        "-",
        "1.0",
        dense_runs,
        dense_ppl,
        achieved_ratio(dense, default="1.0000"),
        "not_applicable",
        dense_notes,
        [length_dir / "dense_full_cache_seed0_generation.json"],
    )

    random_ppls = []
    random_notes = []
    random_ratios = []
    random_gen_paths = []
    for seed in random_seeds:
        path = length_dir / f"spindle_random_seed{seed}_ppl.json"
        data = read_json(path)
        if data and not data.get("_error"):
            random_ppls.append(finite_float(data.get("ppl")))
            ratio = achieved_ratio(data)
            if ratio != "NA":
                random_ratios.append(float(ratio))
        else:
            random_notes.append(f"seed {seed}: {failed_note(path) or (data or {}).get('_error', '')}")
        random_gen_paths.append(length_dir / f"spindle_random_seed{seed}_generation.json")
    add_row(
        max_length,
        "Spindle-Random",
        "spindle",
        "random",
        "-",
        compression_ratio,
        len([value for value in random_ppls if value is not None]),
        mean_std(random_ppls),
        mean_std(random_ratios),
        "not_applicable",
        "strong coverage baseline" if not random_notes else "strong coverage baseline; " + "; ".join(random_notes),
        random_gen_paths,
    )

    variants = [
        (
            "Spindle-SnapKV",
            "mean-1",
            "spindle_snapkv_mean_pool1_seed0_ppl.json",
            "spindle_snapkv_mean_pool1_seed0_generation.json",
            "simplified SnapKV; no local pooling",
        ),
        (
            "Spindle-SnapKV",
            f"mean-{pooled_kernel}",
            f"spindle_snapkv_mean_pool{pooled_kernel}_seed0_ppl.json",
            f"spindle_snapkv_mean_pool{pooled_kernel}_seed0_generation.json",
            "local smoothing",
        ),
        (
            "Spindle-SnapKV",
            f"max-{pooled_kernel}",
            f"spindle_snapkv_max_pool{pooled_kernel}_seed0_ppl.json",
            f"spindle_snapkv_max_pool{pooled_kernel}_seed0_generation.json",
            "local peak saliency",
        ),
        (
            "Spindle-SnapKV",
            f"per-head-{pooled_kernel}",
            f"spindle_snapkv_per_head_pool{pooled_kernel}_seed0_ppl.json",
            f"spindle_snapkv_per_head_pool{pooled_kernel}_seed0_generation.json",
            "closest to full variant",
        ),
    ]

    for method, pooling, ppl_file, gen_file, note in variants:
        path = length_dir / ppl_file
        data = read_json(path)
        if data and not data.get("_error"):
            fallback = bool(data.get("score_method_fallback"))
            ppl_value = finite_float(data.get("ppl"))
            ppl = f"{ppl_value:.4f}" if ppl_value is not None else "nan"
            notes = note
            if fallback:
                notes = f"{note}; fell back to key_norm, not true SnapKV"
            runs = 1
        else:
            ppl = "FAILED"
            notes = f"{note}; {failed_note(path) or (data or {}).get('_error', '')}"
            runs = 0
        add_row(
            max_length,
            method,
            "spindle",
            "snapkv",
            pooling,
            compression_ratio,
            runs,
            ppl,
            achieved_ratio(data),
            snapkv_status(data),
            notes,
            [length_dir / gen_file],
        )

fields = [
    "max_length",
    "method",
    "budget",
    "selection",
    "pooling",
    "ratio",
    "runs",
    "ppl",
    "ttft",
    "tpot",
    "decode_tokens_per_sec",
    "total_tokens_per_sec",
    "peak_cuda_memory_allocated_mb",
    "estimated_kv_cache_memory_mb",
    "achieved_compression_ratio",
    "snapkv_fallback_status",
    "notes",
]

csv_path = root / f"{summary_prefix}_summary.csv"
md_path = root / f"{summary_prefix}_summary.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "| Max Length | Method | Budget | Selection | Pooling | Ratio | Runs | PPL ↓ | TTFT | TPOT | Decode tok/s | Total tok/s | Peak CUDA MB | Est. KV MB | Achieved Ratio | SnapKV Status | Notes |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    lines.append(
        "| "
        + " | ".join(
            str(row[field])
            for field in [
                "max_length",
                "method",
                "budget",
                "selection",
                "pooling",
                "ratio",
                "runs",
                "ppl",
                "ttft",
                "tpot",
                "decode_tokens_per_sec",
                "total_tokens_per_sec",
                "peak_cuda_memory_allocated_mb",
                "estimated_kv_cache_memory_mb",
                "achieved_compression_ratio",
                "snapkv_fallback_status",
                "notes",
            ]
        )
        + " |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def series_label(row):
    if row["selection"] == "none":
        return "Dense"
    if row["selection"] == "random":
        return "Spindle-Random"
    return f"Spindle-SnapKV {row['pooling']}"


def write_svg_plot(metric, title, ylabel, out_path):
    grouped = {}
    for row in rows:
        y = parse_summary_value(row.get(metric))
        if y is None:
            continue
        label = series_label(row)
        grouped.setdefault(label, []).append((int(row["max_length"]), y))
    grouped = {label: sorted(points) for label, points in grouped.items() if points}
    if not grouped:
        out_path.write_text(f"No finite values available for {metric}.\n", encoding="utf-8")
        return

    width, height = 980, 600
    left, right, top, bottom = 90, 240, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({x for points in grouped.values() for x, _ in points})
    ys = [y for points in grouped.values() for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    else:
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

    def sx(x):
        if xmin == xmax:
            return left + plot_w / 2
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#7f7f7f",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="14">Window / prefill length</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 22 {top + plot_h / 2:.1f})">{html.escape(ylabel)}</text>',
    ]

    for x in xs:
        px = sx(x)
        parts.extend([
            f'<line x1="{px:.1f}" y1="{top + plot_h}" x2="{px:.1f}" y2="{top + plot_h + 6}" stroke="#222"/>',
            f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>',
        ])

    for idx in range(6):
        value = ymin + (ymax - ymin) * idx / 5
        py = sy(value)
        parts.extend([
            f'<line x1="{left - 6}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e6e6e6"/>',
            f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3g}</text>',
        ])

    for idx, (label, points) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 24 + idx * 22
        legend_x = left + plot_w + 28
        parts.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>',
        ])

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


plot_titles = {
    "ppl": ("PPL vs Window", "PPL"),
    "decode_tokens_per_sec": ("Decode Tokens/s vs Window", "Decode tokens/s"),
    "estimated_kv_cache_memory_mb": ("Estimated KV Cache Memory vs Window", "Estimated KV cache memory (MB)"),
}
plot_dir = root / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
for metric in plot_metrics:
    title, ylabel = plot_titles.get(metric, (f"{metric} vs Window", metric))
    out = plot_dir / f"{summary_prefix}_{metric}_by_window.svg"
    write_svg_plot(metric, title, ylabel, out)
    print(f"[plot] wrote {out}")

print("\n[summary]")
print(md_path.read_text(encoding="utf-8"))
print(f"[summary] wrote {csv_path}")
print(f"[summary] wrote {md_path}")
PY
