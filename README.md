# PyramidSinkKV for Pythia-70M

This repository implements a training-free KV cache compression method for
`EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped`.  The backbone model
weights are never updated.  All changes happen at inference time by compressing
GPTNeoX/Pythia `past_key_values` after prefill.

The project includes:

- dense generation baseline with no KV compression
- no-cache baseline with `use_cache=False`
- random KV baseline
- uniform SnapKV-style budget
- PyramidSinkKV with random, key-norm, or attention-score token selection
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

Selection methods:

- `random`: reproducible random middle-token selection using `--seed`.
- `key_norm`: selects middle tokens with the largest key-vector L2 norm.
- `attention`: scores historical tokens by recent query attention. If attention
  weights are unavailable, the code logs a warning and falls back to `key_norm`.

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
  --score_method key_norm \
  --max_new_tokens 256 \
  --output_json results/dense_generation.json
```

PyramidSinkKV:

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode pyramid \
  --score_method key_norm \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64 \
  --max_new_tokens 256 \
  --output_json results/pyramid_generation.json
```

Without `--budget_mode` and `--score_method`, the script runs the default method
suite: no-cache, dense, uniform key-norm, uniform random, pyramid key-norm,
pyramid random, reversed key-norm, spindle, hourglass, no-sink, and no-recent.

Reported metrics:

- `TTFT`: time to first token, including prefill and optional cache compression.
- `TPOT`: average time per output token after the first generated token.
- `throughput`: generated tokens per second.
- `total_time`: total generation time.
- `kv_cache_memory_mb`: approximate final KV cache memory.
- `achieved_compression_ratio`: actual compressed cache tokens divided by original cache tokens after prefill.

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
  --score_method key_norm \
  --compression_ratio 0.5 \
  --output_json results/pyramid_wikitext_ppl.json
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
  --score_method key_norm \
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

- `results/ablation_results.json`
- `results/ablation_results.csv`
- `results/ablation_table.md`

The markdown table is intended to be copied directly into the final course
report.  If an experiment fails, the runner records the error in the notes
instead of inventing a result.

### Current Local Run

The table below was generated on this machine with the `pyramidsinkkv` conda
environment, cached `EleutherAI/pythia-70m`, CPU, and float32.  PPL uses
WikiText-2 raw test with `--max_length 512 --stride 64 --max_windows 1`.
Generation uses 128 new tokens. Random baselines report PPL mean/std over seeds
`0,1,2,3,4`; all generation timing columns report mean/std over 3 repeats.

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
/opt/miniconda3/envs/pyramidsinkkv/bin/python -m scripts.run_ablation \
  --model_name_or_path EleutherAI/pythia-70m \
  --compression_ratio 0.5 \
  --max_new_tokens 128 \
  --dataset wikitext \
  --split test \
  --max_length 512 \
  --stride 64 \
  --max_windows 1 \
  --device cpu \
  --dtype float32 \
  --results_dir results/readme_ablation_shapes \
  --random_seeds 0,1,2,3,4 \
  --generation_repeats 3
```

| Method | Budget mode | Score method | Sink size | Recent size | Compression ratio | PPL | TTFT (s) | TPOT (s/token) | Throughput (tok/s) | KV memory | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | no_cache | key_norm | 0 | 0 | 1.0 | 36.8719 | 0.0670 ± 0.0023 | 0.0701 ± 0.0005 | 14.27 ± 0.09 | 0.00 ± 0.00 MB | No KV cache; recomputes full prefix every token |
| dense | dense | key_norm | 4 | 64 | 1.0 | 36.8791 | 0.0642 ± 0.0002 | 0.0045 ± 0.0000 | 201.70 ± 0.98 | 16.50 ± 0.00 MB | Dense baseline |
| random uniform | uniform | random | 4 | 64 | 0.5 | 56.3064 ± 2.5614 | 0.0652 ± 0.0007 | 0.0042 ± 0.0001 | 215.83 ± 2.47 | 9.73 ± 0.00 MB | PPL seeds=0,1,2,3,4 |
| key_norm uniform | uniform | key_norm | 4 | 64 | 0.5 | 74.3237 | 0.0683 ± 0.0013 | 0.0042 ± 0.0001 | 213.33 ± 6.55 | 9.73 ± 0.00 MB | Uniform SnapKV-style budget |
| random pyramid | pyramid | random | 4 | 64 | 0.5 | 65.6226 ± 8.3482 | 0.0680 ± 0.0017 | 0.0043 ± 0.0001 | 210.29 ± 6.96 | 9.78 ± 0.00 MB | PPL seeds=0,1,2,3,4 |
| key_norm pyramid | pyramid | key_norm | 4 | 64 | 0.5 | 72.7206 | 0.0689 ± 0.0015 | 0.0044 ± 0.0001 | 205.71 ± 6.46 | 9.78 ± 0.00 MB | Main PyramidSinkKV variant |
| key_norm reversed | reversed | key_norm | 4 | 64 | 0.5 | 67.6268 | 0.0687 ± 0.0015 | 0.0042 ± 0.0000 | 211.62 ± 1.30 | 9.78 ± 0.00 MB | Reversed-pyramid ablation |
| random spindle | spindle | random | 4 | 64 | 0.5 | 45.6265 ± 3.2682 | 0.0661 ± 0.0011 | 0.0042 ± 0.0000 | 214.17 ± 1.67 | 9.73 ± 0.00 MB | PPL seeds=0,1,2,3,4 |
| key_norm spindle | spindle | key_norm | 4 | 64 | 0.5 | 54.9262 | 0.0671 ± 0.0002 | 0.0042 ± 0.0001 | 211.13 ± 5.36 | 9.73 ± 0.00 MB | Middle layers keep more |
| random hourglass | hourglass | random | 4 | 64 | 0.5 | 77.3285 ± 5.4417 | 0.0674 ± 0.0013 | 0.0043 ± 0.0001 | 210.18 ± 5.26 | 9.74 ± 0.00 MB | PPL seeds=0,1,2,3,4 |
| key_norm hourglass | hourglass | key_norm | 4 | 64 | 0.5 | 85.7068 | 0.0682 ± 0.0015 | 0.0043 ± 0.0001 | 210.45 ± 4.52 | 9.74 ± 0.00 MB | Edge layers keep more |
| pyramid without sink | pyramid | key_norm | 0 | 64 | 0.5 | 107.0948 | 0.0676 ± 0.0010 | 0.0042 ± 0.0001 | 211.74 ± 2.48 | 9.76 ± 0.00 MB | No-sink ablation |
| pyramid without recent | pyramid | key_norm | 4 | 0 | 0.5 | 78.4538 | 0.0677 ± 0.0023 | 0.0042 ± 0.0001 | 212.45 ± 4.06 | 9.74 ± 0.00 MB | No-recent-window ablation |

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
  implementation does not return them, the code falls back to key-norm scoring
  and logs a warning.
