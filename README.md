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
  over batch and query positions, then aggregates heads with
  `--snapkv_head_aggregation mean|max|per_head`, and selects the
  highest-scoring middle tokens for each layer. Sink tokens and recent tokens
  are still always kept.
  Set `--snapkv_pooling_kernel` to an odd value such as `3` or `5` to apply
  length-preserving 1D average pooling over the token scores before top-k
  selection; the default `1` preserves the earlier no-pooling behavior.
  The default head aggregation is `mean`; `max` keeps a token important if any
  single attention head strongly attends to it, while still using one shared
  token index set for all heads.  `per_head` is the closest SnapKV mode: each
  attention head selects and gathers its own middle-token indices.

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

## Running Experiments

Use `scripts.eval_ppl` for continuation perplexity, `scripts.benchmark_generation`
for TTFT/TPOT/throughput, and `scripts.run_ablation` for grouped tables.  The
same core flags can be swapped across scripts:

- Replace `--budget_mode` with `dense`, `no_cache`, `uniform`, `pyramid`,
  `reversed`, `spindle`, or `hourglass`.
- Replace `--score_method` with `random` or `snapkv`.  `key_norm` remains mainly
  an internal fallback when attention weights are unavailable.
- For SnapKV, compare `--snapkv_head_aggregation mean`, `max`, and `per_head`.
  `per_head` is closest to full SnapKV because each head gathers its own indices.
- Use `--snapkv_pooling_kernel 1` to disable pooling, or `3/5` to test local
  score smoothing.
- Use `--compression_ratio 0.75/0.5/0.25` for keep-ratio sweeps.
- Use `--dataset wikitext` for cached quick runs, or `--dataset redpajama
  --redpajama_source hub --redpajama_hub_config wikipedia` for longer text.  The
  RedPajama loader first tries the original JSONL URL list and falls back to the
  HuggingFace converted parquet shard if the raw stream is unstable.
- Use `--device cuda --dtype float16` on GPU, or `--device cpu --dtype float32`
  for reproducible CPU runs.
- For random baselines, repeat with seeds such as `0,1,2,3,4` and report mean
  and standard deviation.  SnapKV is deterministic for a fixed prompt/window.

The main reported metrics are `ppl`, `TTFT`, `TPOT`, `throughput`,
`total_time`, `kv_cache_memory_mb`, `achieved_compression_ratio`,
`scoring_overhead_sec`, `snapkv_fallback_status`, and `notes`.  If an experiment
fails, the ablation runner records the error rather than inventing a result.

Example RedPajama PPL run with spindle budget, full per-head SnapKV, pooling,
and 25% keep ratio:

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
  --budget_mode spindle \
  --score_method snapkv \
  --compression_ratio 0.25 \
  --sink_size 4 \
  --recent_size 64 \
  --observation_window 32 \
  --snapkv_pooling_kernel 3 \
  --snapkv_head_aggregation per_head \
  --device cpu \
  --dtype float32 \
  --output_json results/redpajama_spindle_snapkv_per_head_ppl.json
```

For generation timing, keep the same method flags and switch the module to
`scripts.benchmark_generation`; for grouped experiments, use
`scripts.run_ablation` with `--score_methods random,snapkv` and the desired
comma-separated `--budget_modes`.

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
