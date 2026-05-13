#!/usr/bin/env bash
set -u

# WikiText budget-mode grid for Pythia-70M.
#
# Fixed defaults:
# - dataset: WikiText-2 raw test split
# - compression_ratio: 0.5
# - sink/recent/observation_window: 4 / 64 / 32
# - baseline: dense full KV cache
# - budget modes: uniform, pyramid, reversed, spindle, hourglass
# - score methods: random and snapkv
#
# Example:
#   bash scripts/run_wikitext_budget_grid.sh
#
# Useful overrides:
#   DEVICE=cuda DTYPE=float16 bash scripts/run_wikitext_budget_grid.sh
#   MAX_LENGTH=2048 STRIDE=256 bash scripts/run_wikitext_budget_grid.sh
#   SNAPKV_HEAD_AGGREGATION=mean SNAPKV_POOLING_KERNEL=3 bash scripts/run_wikitext_budget_grid.sh
#   MAX_WINDOWS=1 RANDOM_SEEDS="0" bash scripts/run_wikitext_budget_grid.sh
#   INCLUDE_DENSE_BASELINE=0 bash scripts/run_wikitext_budget_grid.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pyramidsinkkv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-EleutherAI/pythia-70m}"
RESULTS_DIR="${RESULTS_DIR:-results/wikitext_budget_grid_ratio05}"

DATASET="${DATASET:-wikitext}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
STRIDE="${STRIDE:-128}"
MAX_WINDOWS="${MAX_WINDOWS:-8}"

COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
SINK_SIZE="${SINK_SIZE:-4}"
RECENT_SIZE="${RECENT_SIZE:-64}"
OBSERVATION_WINDOW="${OBSERVATION_WINDOW:-32}"
SNAPKV_POOLING_KERNEL="${SNAPKV_POOLING_KERNEL:-3}"
SNAPKV_HEAD_AGGREGATION="${SNAPKV_HEAD_AGGREGATION:-per_head}"

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2 3 4}"
BUDGET_MODES="${BUDGET_MODES:-uniform pyramid reversed spindle hourglass}"
INCLUDE_DENSE_BASELINE="${INCLUDE_DENSE_BASELINE:-1}"

mkdir -p "${RESULTS_DIR}"

echo "[wikitext-grid] results_dir=${RESULTS_DIR}"
echo "[wikitext-grid] dataset=${DATASET} split=${SPLIT}"
echo "[wikitext-grid] budget_modes=${BUDGET_MODES}"
echo "[wikitext-grid] random_seeds=${RANDOM_SEEDS}"
echo "[wikitext-grid] include_dense_baseline=${INCLUDE_DENSE_BASELINE}"
echo "[wikitext-grid] snapkv_head_aggregation=${SNAPKV_HEAD_AGGREGATION}"
echo "[wikitext-grid] snapkv_pooling_kernel=${SNAPKV_POOLING_KERNEL}"
echo "[wikitext-grid] device=${DEVICE} dtype=${DTYPE}"

run_eval() {
  local budget_mode="$1"
  local score_method="$2"
  local seed="$3"
  local output_json="$4"
  local failed_marker="${output_json%.json}.failed.txt"

  if [[ -f "${output_json}" ]]; then
    echo "[skip] ${output_json}"
    return 0
  fi

  echo "[run] budget=${budget_mode} score=${score_method} seed=${seed}"
  "${PYTHON_BIN}" -m scripts.eval_ppl \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_length "${MAX_LENGTH}" \
    --stride "${STRIDE}" \
    --max_windows "${MAX_WINDOWS}" \
    --compression_ratio "${COMPRESSION_RATIO}" \
    --sink_size "${SINK_SIZE}" \
    --recent_size "${RECENT_SIZE}" \
    --observation_window "${OBSERVATION_WINDOW}" \
    --snapkv_pooling_kernel "${SNAPKV_POOLING_KERNEL}" \
    --snapkv_head_aggregation "${SNAPKV_HEAD_AGGREGATION}" \
    --budget_mode "${budget_mode}" \
    --score_method "${score_method}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --output_json "${output_json}"

  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "budget=${budget_mode} score=${score_method} seed=${seed} failed with exit code ${status}" > "${failed_marker}"
    echo "[failed] wrote ${failed_marker}"
  else
    rm -f "${failed_marker}"
  fi
  return 0
}

if [[ "${INCLUDE_DENSE_BASELINE}" != "0" ]]; then
  run_eval \
    "dense" \
    "key_norm" \
    "0" \
    "${RESULTS_DIR}/dense_full_cache_seed0_ppl.json"
fi

for budget_mode in ${BUDGET_MODES}; do
  for seed in ${RANDOM_SEEDS}; do
    run_eval \
      "${budget_mode}" \
      "random" \
      "${seed}" \
      "${RESULTS_DIR}/${budget_mode}_random_seed${seed}_ppl.json"
  done

  run_eval \
    "${budget_mode}" \
    "snapkv" \
    "0" \
    "${RESULTS_DIR}/${budget_mode}_snapkv_${SNAPKV_HEAD_AGGREGATION}_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json"
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${BUDGET_MODES}" <<'PY'
import csv
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
budgets = sys.argv[2].split()
rows = []


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, str(exc)


def failure_note(path):
    marker = Path(str(path).replace(".json", ".failed.txt"))
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    if not path.exists():
        return "missing output json"
    return ""


def achieved_ratio(data):
    if data.get("budget_mode") == "dense":
        return 1.0
    compression = data.get("compression") or []
    if compression and compression[0].get("achieved_compression_ratio") is not None:
        return float(compression[0]["achieved_compression_ratio"])
    return None


for path in sorted(root.glob("dense_full_cache*_ppl.json")):
    data, error = load_json(path)
    if data is not None:
        rows.append({
            "budget_mode": "dense",
            "score_method": "full_cache",
            "runs": 1,
            "ppl": f"{float(data['ppl']):.4f}",
            "ppl_mean": float(data["ppl"]),
            "achieved_compression_ratio": "1.0000",
            "snapkv_fallback_status": "not_applicable",
            "notes": "full KV cache baseline; not memory-matched to compressed runs",
        })
    else:
        rows.append({
            "budget_mode": "dense",
            "score_method": "full_cache",
            "runs": 0,
            "ppl": "FAILED",
            "ppl_mean": float("inf"),
            "achieved_compression_ratio": "1.0000",
            "snapkv_fallback_status": "not_applicable",
            "notes": failure_note(path) or error,
        })

for budget in budgets:
    random_paths = sorted(root.glob(f"{budget}_random_seed*_ppl.json"))
    random_ppls = []
    random_ratios = []
    random_notes = []
    for path in random_paths:
        data, error = load_json(path)
        if data is None:
            random_notes.append(f"{path.name}: {failure_note(path) or error}")
            continue
        random_ppls.append(float(data["ppl"]))
        ratio = achieved_ratio(data)
        if ratio is not None:
            random_ratios.append(ratio)
    if random_ppls:
        rows.append({
            "budget_mode": budget,
            "score_method": "random",
            "runs": len(random_ppls),
            "ppl": f"{st.mean(random_ppls):.4f} +/- {st.pstdev(random_ppls):.4f}" if len(random_ppls) > 1 else f"{random_ppls[0]:.4f}",
            "ppl_mean": st.mean(random_ppls),
            "achieved_compression_ratio": f"{st.mean(random_ratios):.4f}" if random_ratios else "NA",
            "snapkv_fallback_status": "not_applicable",
            "notes": "; ".join(random_notes),
        })

    for path in sorted(root.glob(f"{budget}_snapkv_*_ppl.json")):
        data, error = load_json(path)
        if data is None:
            rows.append({
                "budget_mode": budget,
                "score_method": "snapkv",
                "runs": 0,
                "ppl": "FAILED",
                "ppl_mean": float("inf"),
                "achieved_compression_ratio": "NA",
                "snapkv_fallback_status": "unknown",
                "notes": failure_note(path) or error,
            })
            continue
        fallback = bool(data.get("score_method_fallback"))
        ratio = achieved_ratio(data)
        rows.append({
            "budget_mode": budget,
            "score_method": f"snapkv_{data.get('snapkv_head_aggregation', 'mean')}_pool{data.get('snapkv_pooling_kernel', 1)}",
            "runs": 1,
            "ppl": f"{float(data['ppl']):.4f}",
            "ppl_mean": float(data["ppl"]),
            "achieved_compression_ratio": f"{ratio:.4f}" if ratio is not None else "NA",
            "snapkv_fallback_status": "fallback_to_key_norm" if fallback else "true_snapkv",
            "notes": "not true SnapKV" if fallback else "",
        })

rows.sort(key=lambda item: (item["ppl_mean"], item["budget_mode"], item["score_method"]))

fields = [
    "budget_mode",
    "score_method",
    "runs",
    "ppl",
    "achieved_compression_ratio",
    "snapkv_fallback_status",
    "notes",
]

csv_path = root / "budget_grid_summary.csv"
md_path = root / "budget_grid_summary.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})

lines = [
    "| budget_mode | score_method | runs | ppl | achieved_compression_ratio | snapkv_fallback_status | notes |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n[summary]")
print(md_path.read_text(encoding="utf-8"))
print(f"[summary] wrote {csv_path}")
print(f"[summary] wrote {md_path}")
PY
