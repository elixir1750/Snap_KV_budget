#!/usr/bin/env bash
set -u

# RedPajama-arXiv speed/memory stress benchmark.
#
# This script does not run PPL.  It measures generation speed and memory across
# four prompt window sizes, including long-context stress cases beyond the
# native Pythia-70M 2048 context limit.
#
# Default methods:
# - dense full-cache baseline
# - spindle + random seed 0
# - spindle + snapkv mean pool=7
# - spindle + snapkv max pool=7
# - spindle + snapkv per_head pool=7
#
# Example:
#   bash scripts/run_redpajama_arxiv_speed_memory_windows.sh
#
# Useful overrides:
#   DTYPE=float32 bash scripts/run_redpajama_arxiv_speed_memory_windows.sh
#   LENGTHS="1024 2048" MAX_NEW_TOKENS=256 bash scripts/run_redpajama_arxiv_speed_memory_windows.sh
#   RUN_SNAPKV=0 bash scripts/run_redpajama_arxiv_speed_memory_windows.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pyramidsinkkv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-EleutherAI/pythia-70m}"
RESULTS_DIR="${RESULTS_DIR:-results/redpajama_arxiv_speed_memory_windows}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-arxiv_speed_memory_windows}"

DATASET="${DATASET:-redpajama}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-hub}"
REDPAJAMA_HUB_CONFIG="${REDPAJAMA_HUB_CONFIG:-arxiv}"
SPLIT="${SPLIT:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
LENGTHS="${LENGTHS:-1024 2048 4096 8192}"

COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.25}"
SINK_SIZE="${SINK_SIZE:-4}"
RECENT_SIZE="${RECENT_SIZE:-64}"
OBSERVATION_WINDOW="${OBSERVATION_WINDOW:-32}"
SNAPKV_POOLING_KERNEL="${SNAPKV_POOLING_KERNEL:-7}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0}"
RUN_RANDOM="${RUN_RANDOM:-1}"
RUN_SNAPKV="${RUN_SNAPKV:-1}"

mkdir -p "${RESULTS_DIR}"

echo "[speed-memory] results_dir=${RESULTS_DIR}"
echo "[speed-memory] dataset=${DATASET} redpajama_hub_config=${REDPAJAMA_HUB_CONFIG}"
echo "[speed-memory] lengths=${LENGTHS}"
echo "[speed-memory] max_new_tokens=${MAX_NEW_TOKENS}"
echo "[speed-memory] compression_ratio=${COMPRESSION_RATIO}"
echo "[speed-memory] device=${DEVICE} dtype=${DTYPE}"
echo "[speed-memory] run_random=${RUN_RANDOM} run_snapkv=${RUN_SNAPKV}"
echo "[speed-memory] note: 4096/8192 are speed-memory stress windows for Pythia-70M, not reliable PPL windows."

prompt_dir="${RESULTS_DIR}/prompts"
mkdir -p "${prompt_dir}"
"${PYTHON_BIN}" - "${MODEL_NAME_OR_PATH}" "${prompt_dir}" "${LENGTHS}" "${DATASET}" "${SPLIT}" "${NUM_SAMPLES}" "${REDPAJAMA_SOURCE}" "${REDPAJAMA_HUB_CONFIG}" <<'PY'
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

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
try:
    config = AutoConfig.from_pretrained(model_name)
    max_positions = getattr(config, "max_position_embeddings", None)
except Exception:
    max_positions = None
if max_positions is not None and max(lengths) > int(max_positions):
    print(
        f"[prompt] warning: requested max window {max(lengths)} exceeds "
        f"model max_position_embeddings={max_positions}. Use these rows as speed/memory stress tests."
    )

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

  if [[ -f "${output_json}" ]]; then
    echo "[skip] ${label}: ${output_json}"
    return 0
  fi

  echo "[bench] ${label}: window=${max_length} budget=${budget_mode} score=${score_method} seed=${seed} aggregation=${head_aggregation} pool=${pooling_kernel}"
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
    echo "${label} failed with exit code ${status}" > "${failed_marker}"
    echo "[failed] ${label}; wrote ${failed_marker}"
  else
    rm -f "${failed_marker}"
  fi
  return 0
}

for max_length in ${LENGTHS}; do
  length_dir="${RESULTS_DIR}/window${max_length}"
  mkdir -p "${length_dir}"

  run_benchmark \
    "dense baseline" \
    "${max_length}" \
    "dense" \
    "key_norm" \
    "0" \
    "1" \
    "mean" \
    "${length_dir}/dense_full_cache_seed0_generation.json"

  if [[ "${RUN_RANDOM}" != "0" ]]; then
    for seed in ${RANDOM_SEEDS}; do
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
  fi

  if [[ "${RUN_SNAPKV}" != "0" ]]; then
    run_benchmark \
      "spindle snapkv mean pool=${SNAPKV_POOLING_KERNEL}" \
      "${max_length}" \
      "spindle" \
      "snapkv" \
      "0" \
      "${SNAPKV_POOLING_KERNEL}" \
      "mean" \
      "${length_dir}/spindle_snapkv_mean_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"

    run_benchmark \
      "spindle snapkv max pool=${SNAPKV_POOLING_KERNEL}" \
      "${max_length}" \
      "spindle" \
      "snapkv" \
      "0" \
      "${SNAPKV_POOLING_KERNEL}" \
      "max" \
      "${length_dir}/spindle_snapkv_max_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"

    run_benchmark \
      "spindle snapkv per_head pool=${SNAPKV_POOLING_KERNEL}" \
      "${max_length}" \
      "spindle" \
      "snapkv" \
      "0" \
      "${SNAPKV_POOLING_KERNEL}" \
      "per_head" \
      "${length_dir}/spindle_snapkv_per_head_pool${SNAPKV_POOLING_KERNEL}_seed0_generation.json"
  fi
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${LENGTHS}" "${COMPRESSION_RATIO}" "${RANDOM_SEEDS}" "${SNAPKV_POOLING_KERNEL}" "${SUMMARY_PREFIX}" "${RUN_RANDOM}" "${RUN_SNAPKV}" <<'PY'
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
run_random = sys.argv[7] != "0"
run_snapkv = sys.argv[8] != "0"


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


def fmt(value, precision=4):
    value = finite_float(value)
    return f"{value:.{precision}f}" if value is not None else "NA"


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


def generation_item(path):
    data = read_json(path)
    if not data or data.get("_error"):
        return None
    results = data.get("results") or []
    return results[0] if results else None


def compression_stats(item):
    compression = (item or {}).get("compression") or {}
    return compression if isinstance(compression, dict) else {}


def row_from_item(window, method, budget, selection, pooling, ratio, path, notes):
    item = generation_item(path)
    failed = failed_note(path) if item is None else ""
    comp = compression_stats(item)
    fallback = bool((item or {}).get("score_method_fallback") or comp.get("score_method_fallback"))
    status = "not_applicable"
    if selection == "snapkv":
        status = "fallback_to_key_norm" if fallback else "true_snapkv"
    row = {
        "window_size": window,
        "method": method,
        "budget": budget,
        "selection": selection,
        "pooling": pooling,
        "ratio": ratio,
        "max_new_tokens": (item or {}).get("generated_tokens", "NA"),
        "ttft": fmt((item or {}).get("ttft")),
        "tpot": fmt((item or {}).get("tpot"), precision=6),
        "decode_tokens_per_sec": fmt((item or {}).get("decode_tokens_per_sec")),
        "total_tokens_per_sec": fmt((item or {}).get("total_tokens_per_sec")),
        "peak_cuda_memory_allocated_mb": fmt((item or {}).get("peak_cuda_memory_allocated_mb")),
        "estimated_kv_cache_memory_mb": fmt((item or {}).get("estimated_kv_cache_memory_mb")),
        "achieved_compression_ratio": fmt((item or {}).get("achieved_compression_ratio"), precision=4),
        "snapkv_fallback_status": status,
        "notes": notes if not failed else f"{notes}; {failed}",
    }
    if selection == "snapkv" and fallback:
        row["notes"] = f"{row['notes']}; fell back to key_norm, not true SnapKV"
    return row


rows = []
for window in lengths:
    length_dir = root / f"window{window}"
    rows.append(
        row_from_item(
            window,
            "Dense",
            "full",
            "none",
            "-",
            "1.0",
            length_dir / "dense_full_cache_seed0_generation.json",
            "full KV cache baseline",
        )
    )

    if run_random:
        random_rows = []
        for seed in random_seeds:
            random_rows.append(
                row_from_item(
                    window,
                    "Spindle-Random",
                    "spindle",
                    "random",
                    "-",
                    compression_ratio,
                    length_dir / f"spindle_random_seed{seed}_generation.json",
                    f"random seed {seed}",
                )
            )
        if random_rows:
            merged = dict(random_rows[0])
            merged["notes"] = "random selection baseline"
            merged["max_new_tokens"] = random_rows[0]["max_new_tokens"]
            for key in [
                "ttft",
                "tpot",
                "decode_tokens_per_sec",
                "total_tokens_per_sec",
                "peak_cuda_memory_allocated_mb",
                "estimated_kv_cache_memory_mb",
                "achieved_compression_ratio",
            ]:
                merged[key] = mean_std([parse_summary_value(row[key]) for row in random_rows], precision=6 if key == "tpot" else 4)
            rows.append(merged)

    if run_snapkv:
        variants = [
            (
                "Spindle-SnapKV",
                f"mean-{pooled_kernel}",
                f"spindle_snapkv_mean_pool{pooled_kernel}_seed0_generation.json",
                "local smoothing",
            ),
            (
                "Spindle-SnapKV",
                f"max-{pooled_kernel}",
                f"spindle_snapkv_max_pool{pooled_kernel}_seed0_generation.json",
                "local peak saliency",
            ),
            (
                "Spindle-SnapKV",
                f"per-head-{pooled_kernel}",
                f"spindle_snapkv_per_head_pool{pooled_kernel}_seed0_generation.json",
                "per-head token selection",
            ),
        ]
        for method, pooling, filename, notes in variants:
            rows.append(
                row_from_item(
                    window,
                    method,
                    "spindle",
                    "snapkv",
                    pooling,
                    compression_ratio,
                    length_dir / filename,
                    notes,
                )
            )

fields = [
    "window_size",
    "method",
    "budget",
    "selection",
    "pooling",
    "ratio",
    "max_new_tokens",
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
    "| Window | Method | Budget | Selection | Pooling | Ratio | New Tokens | TTFT | TPOT | Decode tok/s | Total tok/s | Peak CUDA MB | Est. KV MB | Achieved Ratio | SnapKV Status | Notes |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
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
        grouped.setdefault(series_label(row), []).append((int(row["window_size"]), y))
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
        return left + (x - xmin) / (xmax - xmin) * plot_w if xmin != xmax else left + plot_w / 2

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="14">Prompt window size</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 22 {top + plot_h / 2:.1f})">{html.escape(ylabel)}</text>',
    ]
    for x in xs:
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{top + plot_h}" x2="{px:.1f}" y2="{top + plot_h + 6}" stroke="#222"/>')
        parts.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>')
    for idx in range(6):
        value = ymin + (ymax - ymin) * idx / 5
        py = sy(value)
        parts.append(f'<line x1="{left - 6}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3g}</text>')
    for idx, (label, points) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 24 + idx * 22
        legend_x = left + plot_w + 28
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


plot_dir = root / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
plot_specs = [
    ("decode_tokens_per_sec", "Decode Tokens/s vs Window", "Decode tokens/s"),
    ("total_tokens_per_sec", "Total Tokens/s vs Window", "Total tokens/s"),
    ("peak_cuda_memory_allocated_mb", "Peak CUDA Memory vs Window", "Peak CUDA memory (MB)"),
    ("estimated_kv_cache_memory_mb", "Estimated KV Cache Memory vs Window", "Estimated KV cache memory (MB)"),
]
for metric, title, ylabel in plot_specs:
    out = plot_dir / f"{summary_prefix}_{metric}_by_window.svg"
    write_svg_plot(metric, title, ylabel, out)
    print(f"[plot] wrote {out}")

print("\n[summary]")
print(md_path.read_text(encoding="utf-8"))
print(f"[summary] wrote {csv_path}")
print(f"[summary] wrote {md_path}")
PY
