#!/usr/bin/env bash
set -u

# RedPajama budget-mode grid for Pythia-70M.
#
# Fixed defaults:
# - dataset: RedPajama full repo, wikipedia subset
# - compression_ratio: 0.5
# - sink/recent/observation_window: 4 / 64 / 32
# - baseline: dense full KV cache
# - budget modes: uniform, pyramid, reversed, spindle, hourglass
# - score methods: random and snapkv
#
# Override examples:
#   DEVICE=cuda DTYPE=float16 bash scripts/run_redpajama_budget_grid.sh
#   SNAPKV_HEAD_AGGREGATION=mean bash scripts/run_redpajama_budget_grid.sh
#   MAX_WINDOWS=1 NUM_SAMPLES=2 bash scripts/run_redpajama_budget_grid.sh
#   INCLUDE_DENSE_BASELINE=0 bash scripts/run_redpajama_budget_grid.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pyramidsinkkv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-EleutherAI/pythia-70m}"
RESULTS_DIR="${RESULTS_DIR:-results/redpajama_budget_grid_ratio05}"

DATASET="${DATASET:-redpajama}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-hub}"
REDPAJAMA_HUB_CONFIG="${REDPAJAMA_HUB_CONFIG:-wikipedia}"
SPLIT="${SPLIT:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
STRIDE="${STRIDE:-256}"
MAX_WINDOWS="${MAX_WINDOWS:-2}"

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

echo "[grid] results_dir=${RESULTS_DIR}"
echo "[grid] budget_modes=${BUDGET_MODES}"
echo "[grid] random_seeds=${RANDOM_SEEDS}"
echo "[grid] include_dense_baseline=${INCLUDE_DENSE_BASELINE}"
echo "[grid] snapkv_head_aggregation=${SNAPKV_HEAD_AGGREGATION}"
echo "[grid] device=${DEVICE} dtype=${DTYPE}"

run_eval() {
  local budget_mode="$1"
  local score_method="$2"
  local seed="$3"
  local output_json="$4"

  if [[ -f "${output_json}" ]]; then
    echo "[skip] ${output_json}"
    return 0
  fi

  echo "[run] budget=${budget_mode} score=${score_method} seed=${seed}"
  "${PYTHON_BIN}" -m scripts.eval_ppl \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset "${DATASET}" \
    --redpajama_source "${REDPAJAMA_SOURCE}" \
    --redpajama_hub_config "${REDPAJAMA_HUB_CONFIG}" \
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
}

if [[ "${INCLUDE_DENSE_BASELINE}" != "0" ]]; then
  run_eval \
    "dense" \
    "key_norm" \
    "0" \
    "${RESULTS_DIR}/dense_full_cache_seed0_ppl.json" || true
fi

for budget_mode in ${BUDGET_MODES}; do
  for seed in ${RANDOM_SEEDS}; do
    run_eval \
      "${budget_mode}" \
      "random" \
      "${seed}" \
      "${RESULTS_DIR}/${budget_mode}_random_seed${seed}_ppl.json" || true
  done

  run_eval \
    "${budget_mode}" \
    "snapkv" \
    "0" \
    "${RESULTS_DIR}/${budget_mode}_snapkv_${SNAPKV_HEAD_AGGREGATION}_pool${SNAPKV_POOLING_KERNEL}_seed0_ppl.json" || true
done

"${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY'
import csv
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
budgets = ["uniform", "pyramid", "reversed", "spindle", "hourglass"]
rows = []

def achieved_ratio(data):
    if data.get("budget_mode") == "dense":
        return 1.0
    compression = data.get("compression") or []
    if compression and compression[0].get("achieved_compression_ratio") is not None:
        return float(compression[0]["achieved_compression_ratio"])
    return None

for budget in budgets:
    random_paths = sorted(root.glob(f"{budget}_random_seed*_ppl.json"))
    random_ppls = []
    random_ratios = []
    random_notes = []
    for path in random_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            random_ppls.append(float(data["ppl"]))
            ratio = achieved_ratio(data)
            if ratio is not None:
                random_ratios.append(ratio)
        except Exception as exc:
            random_notes.append(f"{path.name}: {exc}")
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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fallback = bool(data.get("score_method_fallback"))
            rows.append({
                "budget_mode": budget,
                "score_method": f"snapkv_{data.get('snapkv_head_aggregation', 'mean')}_pool{data.get('snapkv_pooling_kernel', 1)}",
                "runs": 1,
                "ppl": f"{float(data['ppl']):.4f}",
                "ppl_mean": float(data["ppl"]),
                "achieved_compression_ratio": f"{achieved_ratio(data):.4f}" if achieved_ratio(data) is not None else "NA",
                "snapkv_fallback_status": "fallback_to_key_norm" if fallback else "true_snapkv",
                "notes": "not true SnapKV" if fallback else "",
            })
        except Exception as exc:
            rows.append({
                "budget_mode": budget,
                "score_method": "snapkv",
                "runs": 0,
                "ppl": "FAILED",
                "ppl_mean": float("inf"),
                "achieved_compression_ratio": "NA",
                "snapkv_fallback_status": "unknown",
                "notes": f"{path.name}: {exc}",
            })

for path in sorted(root.glob("dense_full_cache*_ppl.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
    except Exception as exc:
        rows.append({
            "budget_mode": "dense",
            "score_method": "full_cache",
            "runs": 0,
            "ppl": "FAILED",
            "ppl_mean": float("inf"),
            "achieved_compression_ratio": "1.0000",
            "snapkv_fallback_status": "not_applicable",
            "notes": f"{path.name}: {exc}",
        })

rows.sort(key=lambda item: (item["ppl_mean"], item["budget_mode"], item["score_method"]))

csv_path = root / "budget_grid_summary.csv"
md_path = root / "budget_grid_summary.md"
fields = [
    "budget_mode",
    "score_method",
    "runs",
    "ppl",
    "achieved_compression_ratio",
    "snapkv_fallback_status",
    "notes",
]

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
