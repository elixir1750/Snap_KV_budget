#!/usr/bin/env bash
set -u

# WikiText short-context spindle + SnapKV variant experiment.
#
# This wrapper reuses the RedPajama/arXiv length-ablation driver, but switches
# the data source and output names to WikiText.  It keeps the same method set:
# dense, spindle+random seeds, and spindle+SnapKV mean/max/per-head variants.
#
# Example:
#   bash scripts/run_wikitext_spindle_snapkv_length_ablation.sh
#
# Useful overrides:
#   DEVICE=cpu DTYPE=float32 NUM_SAMPLES=1 MAX_WINDOWS=1 bash scripts/run_wikitext_spindle_snapkv_length_ablation.sh
#   LENGTHS="512 1024" bash scripts/run_wikitext_spindle_snapkv_length_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-wikitext}"
export SPLIT="${SPLIT:-test}"
export RESULTS_DIR="${RESULTS_DIR:-results/wikitext_spindle_snapkv_length_ablation_ratio025}"
export SUMMARY_PREFIX="${SUMMARY_PREFIX:-wikitext_length_ablation}"
export LENGTHS="${LENGTHS:-256 512 1024 1536}"
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export MAX_WINDOWS="${MAX_WINDOWS:-8}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bfloat16}"
export RUN_GENERATION_BENCHMARKS="${RUN_GENERATION_BENCHMARKS:-0}"
export PLOT_METRICS="${PLOT_METRICS:-ppl}"

bash "${SCRIPT_DIR}/run_redpajama_arxiv_length_ablation.sh"
