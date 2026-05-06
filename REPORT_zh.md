# PyramidSinkKV + SnapKV 中文报告

## 项目背景

本项目面向 `EleutherAI/pythia-70m` / GPTNeoX 结构，实现训练无关的 KV cache 压缩。模型权重不更新，推理时只改变 prefill 后 `past_key_values` 中保留哪些历史 token，从而研究上下文质量、KV 显存占用和生成速度之间的折中。

## 为什么把 KV 压缩拆成两个问题

KV 压缩可以拆成两个相对独立的问题：

1. 每层保留多少 token：由 `budget_mode` 控制。
2. 每层保留哪些 token：由 `score_method` 控制。

这样做的好处是可以公平比较不同 layer-wise budget 形状。`uniform`、`pyramid`、`reversed`、`spindle`、`hourglass` 在同一个 `compression_ratio` 下会尽量保持相近的平均保留比例，避免某个方法因为总 KV budget 更多而占便宜。

## SnapKV 的作用

新增的 `score_method=snapkv` 是 attention-based token selection。对于每一层：

- 始终保留 sink tokens。
- 始终保留 recent tokens。
- 对中间区域 token，用 prefill 阶段返回的 attention weights 计算重要性。
- 取最后 `observation_window` 个 query 位置到所有历史 key 位置的注意力：

```text
recent_attn = attentions[layer][:, :, -observation_window:, :]
score = recent_attn.mean(dim=(0, 1, 2))
```

得到形状为 `[seq_len]` 的分数后，只在中间区域做 top-k，最后将 sink、SnapKV middle、recent 合并去重并按原始位置排序，以保证压缩后的 K/V 仍保持原始 token 顺序。

## Layer-wise Budget 的作用

- `uniform`：每层保留相同 token 比例。
- `pyramid`：底层保留更多，高层保留更少。
- `reversed`：底层保留更少，高层保留更多。
- `spindle`：中间层保留更多，两端层保留更少。
- `hourglass`：两端层保留更多，中间层保留更少。

这些策略只决定“每层保留多少 token”。在本项目中，`snapkv` 决定“中间区域具体保留哪些 token”。

## 实现细节

核心实现位于 `pyramidsinkkv.py`：

- `PyramidSinkKVConfig` 增加 `observation_window`、`debug_selection` 等配置。
- `_snapkv_attention_scores` 显式处理 `[batch, heads, query_len, key_len]` attention tensor，并返回 `[seq_len]` 分数。
- `select_token_indices` 保证 selected indices 为一维、递增、唯一、合法。
- `gather_kv_by_indices` 只沿 seq 维度 gather K/V。
- `compress_past_key_values` 记录 `achieved_compression_ratio`、`per_layer_keep_lengths`、fallback 状态等诊断信息。

SnapKV 需要 attention weights。加载模型时会优先尝试 `attn_implementation="eager"`；如果当前 Transformers/GPTNeoX 组合不支持，会重试普通加载。如果 prefill 后仍然拿不到 attention weights，会明确记录 `score_method_fallback=true`，并回退到 `key_norm`，不会静默改变实验设置。

## RoPE / position_ids 注意事项

GPTNeoX/Pythia 使用 RoPE。KV cache 被压缩后，物理 cache 长度会小于真实上下文长度，但下一个 token 的 `position_ids` 必须继续使用真实绝对位置。

例如 prefill 长度为 1024，压缩后 KV 长度为 512，下一个生成 token 仍然必须使用：

```text
position_ids = 1024
cache_position = 1024
```

因此代码维护 `logical_seq_len`。解码时 attention mask 对应压缩后的物理 KV 长度加当前 token，而 RoPE 位置仍然来自真实逻辑位置。`compressed_cache_len != logical_seq_len` 是预期情况。

## 实验设置

推荐命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
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

输出文件包括：

- `results/group_main_ablation.{json,csv,md}`
- `results/group_ratio_sweep.{json,csv,md}`
- `results/group_sink_recent_ablation.{json,csv,md}`

表格指标包括 `ppl`、`ttft`、`tpot`、`throughput`、`total_time`、`kv_cache_memory_mb`、`achieved_compression_ratio`、`scoring_overhead_sec`、`snapkv_fallback_status` 和 `notes`。如果 `score_method=snapkv` 回退到 `key_norm`，该行必须标记为 `fallback_to_key_norm`，并且不能解释为真正的 SnapKV 结果。

## 主实验表格

见 `GROUP_REPORT_TABLES_zh.md`。如果尚未运行真实实验，表格保留 TODO，不填入虚构数字。

## Compression Ratio Sweep 表格

见 `GROUP_REPORT_TABLES_zh.md`。计划比较 `compression_ratio` 为 0.75、0.5、0.25 时，`uniform/spindle + random` 和 `uniform/spindle + snapkv` 的表现，并只保留一次 dense baseline。

## Sink/Recent 消融表格

见 `GROUP_REPORT_TABLES_zh.md`。默认使用 `spindle + snapkv`，比较保留 sink+recent、去掉 sink、去掉 recent、两者都去掉。

## 结果分析

本次 CPU/float32 小样本实验中，dense PPL 为 21.1162，no-cache PPL 基本相同，但 TPOT 从 dense 的 0.0049 s/token 上升到 0.0854 s/token，说明 KV cache 对自回归解码速度仍然非常关键。

在 `compression_ratio=0.5` 的主实验中，spindle 系列明显优于 uniform 和 pyramid。`spindle + random` 的 PPL 为 26.3768 +/- 2.1393，`spindle + snapkv` 的 PPL 为 26.4305，二者非常接近，并且都显著优于 `uniform + random` 的 33.8358 和 `uniform + snapkv` 的 40.3360。`hourglass + snapkv` 的 PPL 为 58.8478，是主实验中较差的压缩策略，说明把预算偏向两端层不适合这个设置。

ratio sweep 中，压缩越强 PPL 越高：例如 `uniform + snapkv` 从 ratio 0.75 的 29.5593 上升到 ratio 0.25 的 58.2325；`spindle + snapkv` 从 23.8238 上升到 47.7264。spindle 在三个 ratio 下都比 uniform 更稳。random baseline 在这个短窗口 WikiText 设置下仍然很强，尤其 `spindle + random` 在 ratio 0.75/0.5/0.25 下分别为 22.5578、26.3768、41.0463，均不差于对应 SnapKV。

sink/recent 消融在本次 `spindle + snapkv` 设置下差异很小：默认 sink+recent 的 PPL 为 26.4305，去掉 sink 为 26.2927，去掉 recent 为 26.2726，两者都去掉为 26.2553。这个现象只说明当前两窗口小样本不敏感，不能推广到长上下文或更大模型。

所有 SnapKV 行的 `snapkv_fallback_status` 均为 `true_snapkv`，没有发生 key_norm fallback。需要注意的是，SnapKV prefill 为了获取 attention weights 使用 eager attention；压缩后 decode 切回 SDPA，以支持不同层有不同压缩后 KV 长度的 layer-wise budget。

## 局限性

- 获取 attention weights 会带来 TTFT overhead；当前实现是研究型、易复现版本，不是 kernel-fused 优化版本。
- Pythia-70M 模型太小，速度收益可能不明显。
- CPU timing 不代表 GPU kernel-level speedup。
- random baseline 可能在短上下文 WikiText 上较强，单次小样本实验不能过度解读。

## 结论

本项目现在把 layer-wise budget 和 token selection 清晰解耦：`budget_mode` 控制每层保留多少 token，`score_method=snapkv` 使用 attention-based 分数决定中间区域保留哪些 token。主实验聚焦 dense/no-cache、random 和 SnapKV；`key_norm` 只作为 SnapKV attention weights 不可用时的内部 fallback，并在表格中通过 `snapkv_fallback_status` 明确标出。
