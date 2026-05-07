# Pythia-70M KV Cache Budgeting

This repository studies training-free inference-time KV cache budgeting and
compression for `EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped`.
The backbone model weights are never updated.  All changes happen at inference
time by changing how GPTNeoX/Pythia `past_key_values` are kept after prefill.

The project includes:

- dense generation baseline with no KV compression
- no-cache baseline with `use_cache=False`
- random KV baseline
- uniform, pyramid, reversed, spindle, and hourglass layer-wise KV budgets
- layer-budgeted KV compression with random or SnapKV token selection
- reversed-pyramid, spindle, hourglass, no-sink, and no-recent-window ablations
- generation speed benchmark, PPL evaluation, ablation table generation, and a terminal streaming demo

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code uses HuggingFace APIs directly, so mirror settings are handled through
standard environment variables.  For example:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then load either:

```bash
EleutherAI/pythia-70m
EleutherAI/pythia-70m-deduped
```

## Method

`pyramidsinkkv.py` contains the core implementation.  For each layer, it computes
a keep budget, always preserves sink tokens and recent tokens, selects remaining
middle tokens, and gathers KV tensors in their original order.

Expected GPTNeoX/Pythia cache tensor shape:

```text
[batch, num_heads, seq_len, head_dim]
```

Budget modes:

- `uniform`: every layer keeps the same ratio.
- `pyramid`: lower layers keep more tokens and higher layers keep fewer.
- `reversed`: lower layers keep fewer tokens and higher layers keep more.
- `spindle`: middle layers keep more tokens and edge layers keep fewer.
- `hourglass`: edge layers keep more tokens and middle layers keep fewer.
- `dense`: disables compression for evaluation scripts.
- `no_cache`: disables KV cache entirely and recomputes the full prefix every token.

`budget_mode` and `score_method` intentionally answer two different questions:
`budget_mode` decides how many tokens each layer keeps, while `score_method`
decides which middle tokens are kept after sink and recent tokens are reserved.

Selection methods:

- `random`: reproducible random middle-token selection using `--seed`.
- `key_norm`: internal fallback only; it selects middle tokens with the largest
  key-vector L2 norm when attention weights are unavailable.
- `attention`: scores historical tokens by recent query attention. If attention
  weights are unavailable, the code logs a warning and falls back to `key_norm`.
- `snapkv`: recommended attention-based method. During prefill it reads
  attention weights, averages the last `--observation_window` query positions
  over batch, heads, and query positions, and selects the highest-scoring middle
  tokens for each layer. Sink tokens and recent tokens are still always kept.
  Set `--snapkv_pooling_kernel` to an odd value such as `3` or `5` to apply
  length-preserving 1D average pooling over the token scores before top-k
  selection; the default `1` preserves the earlier no-pooling behavior.

SnapKV requires attention weights. The loader requests eager attention when
possible via `attn_implementation="eager"` and retries without it on older
Transformers/model combinations. If attention weights are still unavailable,
the result records `score_method_fallback=true` and uses `key_norm` rather than
silently changing behavior. Returning attention weights may increase TTFT; this
implementation favors correctness and reproducibility over fused-kernel speed.

## RoPE Correctness

GPTNeoX/Pythia uses RoPE, so compressed cache length must not be treated as the
true token position.  The manual generation loop keeps a `logical_seq_len`
counter.  After prefill, the KV cache may be compressed from, for example, 1024
tokens to 512 tokens, but the next generated token still receives
`position_ids=1024` and `cache_position=1024`.

This is why the scripts use `pyramidsinkkv.generate(...)` instead of relying on
the default `model.generate(...)` cache-length inference.  The smoke test in
`tests/test_pyramidsinkkv.py` also checks that compressed cache length and
logical sequence length can differ.

## Generation Benchmark

Dense baseline:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode dense \
  --score_method random \
  --max_new_tokens 256 \
  --output_json results/dense_generation.json
```

PyramidSinkKV:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode pyramid \
  --score_method random \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --max_new_tokens 256 \
  --output_json results/pyramid_generation.json
```

Uniform SnapKV:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode uniform \
  --score_method snapkv \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --max_new_tokens 256 \
  --output_json results/uniform_snapkv_generation.json
```

Pyramid SnapKV:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode pyramid \
  --score_method snapkv \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --max_new_tokens 256 \
  --output_json results/pyramid_snapkv_generation.json
```

Spindle SnapKV:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode spindle \
  --score_method snapkv \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --max_new_tokens 256 \
  --output_json results/spindle_snapkv_generation.json
```

Without `--budget_mode` and `--score_method`, the script runs a compact method
suite focused on dense/no-cache baselines, random token selection, SnapKV
attention-based selection, and the supported layer-wise budget modes.

Reported metrics:

- `TTFT`: time to first token, including prefill and optional cache compression.
- `TPOT`: average time per output token after the first generated token.
- `throughput`: generated tokens per second.
- `total_time`: total generation time.
- `scoring_overhead_sec`: extra scoring pass overhead when present. Current
  SnapKV uses attention weights returned by prefill, so the attention-return
  cost is included in TTFT.
- `compression_overhead_sec`: time spent selecting indices and gathering K/V.
- `kv_cache_memory_mb`: approximate final KV cache memory.
- `achieved_compression_ratio`: actual compressed cache tokens divided by original cache tokens after prefill.
- `snapkv_fallback_status`: whether SnapKV was true attention-based selection
  or fell back to `key_norm`.

Use longer generation lengths such as 256 or 512 tokens; Pythia-70M is small, so
very short generations can hide speed differences.

## Perplexity Evaluation

WikiText:

```bash
python -m scripts.eval_ppl \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset wikitext \
  --split test \
  --max_length 1024 \
  --stride 128 \
  --budget_mode pyramid \
  --score_method random \
  --compression_ratio 0.5 \
  --output_json results/pyramid_wikitext_ppl.json
```

WikiText with spindle + SnapKV:

```bash
python -m scripts.eval_ppl \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset wikitext \
  --split test \
  --max_length 1024 \
  --stride 128 \
  --budget_mode spindle \
  --score_method snapkv \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --output_json results/spindle_snapkv_wikitext_ppl.json
```

PG-19 single-sample quick experiment:

```bash
python -m scripts.eval_ppl \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset pg19 \
  --split test \
  --num_samples 1 \
  --max_windows 2 \
  --max_length 1024 \
  --stride 128 \
  --budget_mode pyramid \
  --score_method random \
  --compression_ratio 0.5 \
  --output_json results/pyramid_pg19_ppl.json
```

For dense PPL, use `--budget_mode dense`.

## Ablation Table

Run the full small ablation suite:

```bash
python -m scripts.run_ablation \
  --model_name_or_path EleutherAI/pythia-70m \
  --compression_ratio 0.5 \
  --max_new_tokens 256 \
  --max_windows 2
```

Outputs:

- `results/group_main_ablation.json`
- `results/group_main_ablation.csv`
- `results/group_main_ablation.md`
- `results/group_ratio_sweep.json`
- `results/group_ratio_sweep.csv`
- `results/group_ratio_sweep.md`
- `results/group_sink_recent_ablation.json`
- `results/group_sink_recent_ablation.csv`
- `results/group_sink_recent_ablation.md`

SnapKV group ablation:

```bash
python -m scripts.run_ablation \
  --model_name_or_path EleutherAI/pythia-70m \
  --compression_ratio 0.5 \
  --max_new_tokens 256 \
  --dataset wikitext \
  --split test \
  --max_length 1024 \
  --stride 128 \
  --max_windows 2 \
  --score_methods random,snapkv \
  --budget_modes uniform,pyramid,reversed,spindle,hourglass
```

## GPU Experiment Workflow

Use the `pyramidsinkkv` conda environment and confirm CUDA is visible:

```bash
conda activate pyramidsinkkv
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

For HuggingFace access in China, the mirror can be useful for model files:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

If a public dataset reports `Invalid username or password`, clear stale tokens:

```bash
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN
```

Run the grouped WikiText ablation on GPU:

```bash
python -m scripts.run_ablation \
  --model_name_or_path EleutherAI/pythia-70m \
  --compression_ratio 0.5 \
  --max_new_tokens 256 \
  --dataset wikitext \
  --split test \
  --max_length 1024 \
  --stride 128 \
  --max_windows 2 \
  --score_methods random,snapkv \
  --budget_modes uniform,pyramid,reversed,spindle,hourglass \
  --device cuda \
  --dtype float16 \
  --generation_repeats 5 \
  --results_dir results/gpu_wikitext_ablation
```

Run the longer RedPajama probe on GPU without executing the legacy dataset
script:

```bash
python -m scripts.eval_ppl \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset redpajama \
  --redpajama_source hub \
  --redpajama_hub_config wikipedia \
  --split train \
  --num_samples 8 \
  --max_length 2048 \
  --stride 256 \
  --max_windows 2 \
  --compression_ratio 0.25 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --budget_mode spindle \
  --score_method snapkv \
  --seed 0 \
  --device cuda \
  --dtype float16 \
  --output_json results/gpu_redpajama_spindle_025_len2048_stride256/spindle_snapkv_seed0_ppl.json
```

For the matching random baseline, repeat the RedPajama command with
`--score_method random` and seeds such as `0,1,2,3,4`, then report mean and
standard deviation for PPL.  CUDA timings use `torch.cuda.synchronize()` in the
benchmark script; `eval_ppl.py` reports PPL rather than generation speed.

The markdown table is intended to be copied directly into the final course
report.  If an experiment fails, the runner records the error in the notes
instead of inventing a result.

## Terminal Speed Demo

```bash
python -m scripts.demo_speed_animation \
  --model_name_or_path EleutherAI/pythia-70m \
  --methods dense,pyramid \
  --max_new_tokens 256 \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64
```

If `rich` is installed, the demo shows a live panel.  Otherwise it falls back to
plain stdout refresh.

## Smoke Tests

```bash
python -m pytest tests
```

or, without pytest:

```bash
python tests/test_pyramidsinkkv.py
python -m compileall pyramidsinkkv.py scripts tests
```

## Notes

- This is a training-free method: no trainable parameters, no finetuning, and no
  model weight updates.
- The initial implementation compresses after prefill.  Repeated compression
  during decoding is intentionally left out to keep the course-project baseline
  reliable and easy to inspect.
- Attention-score selection requires attention weights.  If the active attention
  implementation does not return them, the code falls back to key-norm scoring,
  logs a warning, and records the fallback in JSON/CSV/markdown outputs.
