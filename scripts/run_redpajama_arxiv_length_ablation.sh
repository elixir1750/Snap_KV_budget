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
#   DEVICE=cuda DTYPE=float16 bash scripts/run_redpajama_arxiv_length_ablation.sh
#   LENGTHS="2048 4096" MAX_WINDOWS=1 bash scripts/run_redpajama_arxiv_length_ablation.sh
#   RANDOM_SEEDS="0" NUM_SAMPLES=2 bash scripts/run_redpajama_arxiv_length_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pyramidsinkkv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-EleutherAI/pythia-70m}"
RESULTS_DIR="${RESULTS_DIR:-results/redpajama_arxiv_spindle_length_ablation_ratio025}"

DATASET="${DATASET:-redpajama}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-hub}"
REDPAJAMA_HUB_CONFIG="${REDPAJAMA_HUB_CONFIG:-arxiv}"
SPLIT="${SPLIT:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
LENGTHS="${LENGTHS:-2048 4096 6144 8192}"
STRIDE="${STRIDE:-256}"
MAX_WINDOWS="${MAX_WINDOWS:-2}"

COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.25}"
SINK_SIZE="${SINK_SIZE:-4}"
RECENT_SIZE="${RECENT_SIZE:-64}"
OBSERVATION_WINDOW="${OBSERVATION_WINDOW:-32}"
SNAPKV_POOLING_KERNEL="${SNAPKV_POOLING_KERNEL:-7}"

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2 3 4}"

mkdir -p "${RESULTS_DIR}"

echo "[arxiv-length] results_dir=${RESULTS_DIR}"
echo "[arxiv-length] redpajama_hub_config=${REDPAJAMA_HUB_CONFIG}"
echo "[arxiv-length] lengths=${LENGTHS}"
echo "[arxiv-length] compression_ratio=${COMPRESSION_RATIO}"
echo "[arxiv-length] observation_window=${OBSERVATION_WINDOW}"
echo "[arxiv-length] pooled_snapkv_kernel=${SNAPKV_POOLING_KERNEL}"
echo "[arxiv-length] random_seeds=${RANDOM_SEEDS}"
echo "[arxiv-length] device=${DEVICE} dtype=${DTYPE}"

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

  run_eval \
    "spindle snapkv mean pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "mean" \
    "${length_dir}/spindle_snapkv_mean_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"

  run_eval \
    "spindle snapkv max pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "max" \
    "${length_dir}/spindle_snapkv_max_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"

  run_eval \
    "spindle snapkv per_head pool=${SNAPKV_POOLING_KERNEL}" \
    "${max_length}" \
    "spindle" \
    "snapkv" \
    "0" \
    "${SNAPKV_POOLING_KERNEL}" \
    "per_head" \
    "${length_dir}/spindle_snapkv_per_head_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${LENGTHS}" "${COMPRESSION_RATIO}" "${RANDOM_SEEDS}" "${SNAPKV_POOLING_KERNEL}" <<'PY'
import csv
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
lengths = [int(item) for item in sys.argv[2].split()]
compression_ratio = sys.argv[3]
random_seeds = sys.argv[4].split()
pooled_kernel = sys.argv[5]


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


rows = []

for max_length in lengths:
    length_dir = root / f"len{max_length}"

    dense_path = length_dir / "dense_full_cache_seed0_ppl.json"
    dense = read_json(dense_path)
    if dense and not dense.get("_error"):
        dense_ppl = f"{float(dense['ppl']):.4f}"
        dense_notes = "baseline"
    else:
        dense_ppl = "FAILED"
        dense_notes = failed_note(dense_path) or (dense or {}).get("_error", "")
    rows.append({
        "max_length": max_length,
        "method": "Dense",
        "budget": "full",
        "selection": "none",
        "pooling": "-",
        "ratio": "1.0",
        "runs": 1 if dense_ppl != "FAILED" else 0,
        "ppl": dense_ppl,
        "achieved_compression_ratio": achieved_ratio(dense, default="1.0000"),
        "snapkv_fallback_status": "not_applicable",
        "notes": dense_notes,
    })

    random_ppls = []
    random_notes = []
    random_ratios = []
    for seed in random_seeds:
        path = length_dir / f"spindle_random_seed{seed}_ppl.json"
        data = read_json(path)
        if data and not data.get("_error"):
            random_ppls.append(float(data["ppl"]))
            ratio = achieved_ratio(data)
            if ratio != "NA":
                random_ratios.append(float(ratio))
        else:
            random_notes.append(f"seed {seed}: {failed_note(path) or (data or {}).get('_error', '')}")

    if random_ppls:
        random_ppl = f"{st.mean(random_ppls):.4f} +/- {st.pstdev(random_ppls):.4f}"
        random_ratio = f"{st.mean(random_ratios):.4f}" if random_ratios else "NA"
        random_runs = len(random_ppls)
    else:
        random_ppl = "FAILED"
        random_ratio = "NA"
        random_runs = 0
    rows.append({
        "max_length": max_length,
        "method": "Spindle-Random",
        "budget": "spindle",
        "selection": "random",
        "pooling": "-",
        "ratio": compression_ratio,
        "runs": random_runs,
        "ppl": random_ppl,
        "achieved_compression_ratio": random_ratio,
        "snapkv_fallback_status": "not_applicable",
        "notes": "strong coverage baseline" if not random_notes else "strong coverage baseline; " + "; ".join(random_notes),
    })

    variants = [
        ("Spindle-SnapKV", "mean-1", "spindle_snapkv_mean_pool1_seed0_ppl.json", "simplified SnapKV; no local pooling"),
        ("Spindle-SnapKV", f"mean-{pooled_kernel}", f"spindle_snapkv_mean_pool{pooled_kernel}_seed0_ppl.json", "local smoothing"),
        ("Spindle-SnapKV", f"max-{pooled_kernel}", f"spindle_snapkv_max_pool{pooled_kernel}_seed0_ppl.json", "local peak saliency"),
        ("Spindle-SnapKV", f"per-head-{pooled_kernel}", f"spindle_snapkv_per_head_pool{pooled_kernel}_seed0_ppl.json", "closest to full variant"),
    ]

    for method, pooling, filename, note in variants:
        path = length_dir / filename
        data = read_json(path)
        if data and not data.get("_error"):
            fallback = bool(data.get("score_method_fallback"))
            ppl = f"{float(data['ppl']):.4f}"
            notes = note
            if fallback:
                notes = f"{note}; fell back to key_norm, not true SnapKV"
            runs = 1
        else:
            ppl = "FAILED"
            notes = f"{note}; {failed_note(path) or (data or {}).get('_error', '')}"
            runs = 0
        rows.append({
            "max_length": max_length,
            "method": method,
            "budget": "spindle",
            "selection": "snapkv",
            "pooling": pooling,
            "ratio": compression_ratio,
            "runs": runs,
            "ppl": ppl,
            "achieved_compression_ratio": achieved_ratio(data),
            "snapkv_fallback_status": snapkv_status(data),
            "notes": notes,
        })

fields = [
    "max_length",
    "method",
    "budget",
    "selection",
    "pooling",
    "ratio",
    "runs",
    "ppl",
    "achieved_compression_ratio",
    "snapkv_fallback_status",
    "notes",
]

csv_path = root / "arxiv_length_ablation_summary.csv"
md_path = root / "arxiv_length_ablation_summary.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "| Max Length | Method | Budget | Selection | Pooling | Ratio | Runs | PPL ↓ | Achieved Ratio | SnapKV Status | Notes |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                "achieved_compression_ratio",
                "snapkv_fallback_status",
                "notes",
            ]
        )
        + " |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n[summary]")
print(md_path.read_text(encoding="utf-8"))
print(f"[summary] wrote {csv_path}")
print(f"[summary] wrote {md_path}")
PY
