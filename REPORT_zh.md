# PyramidSinkKV for Pythia-70M 实验报告

## 项目背景

本项目关注语言模型的训练免费高效推理。目标模型是
`EleutherAI/pythia-70m` 或 `EleutherAI/pythia-70m-deduped`，方法只改变推理
阶段的 KV cache 保存策略，不更新模型权重，也不引入可训练参数。

Pythia-70M 规模较小，适合作为课程项目中的可复现实验对象。虽然它本身显存压力
不大，但 KV cache 压缩能够清楚展示长上下文推理中“保留哪些历史 token”这个核心
问题。

## 方法简介

PyramidSinkKV 在 prefill 后对每一层的 KV cache 做一次压缩。对于每层缓存，方法会：

1. 保留开头的 sink tokens；
2. 保留最近的 recent window tokens；
3. 在中间区域按指定分数选择 token；
4. 将选中的索引排序，保持原始 token 顺序；
5. 用这些索引 gather 每一层的 key/value 张量。

支持的中间 token 选择方式包括 `random`、`key_norm` 和 `attention`。其中
`attention` 需要模型实际返回 attention weights；如果当前 attention 实现不返回，
代码会自动回退到 `key_norm` 并记录 warning。

## 为什么选择 Pyramid + SnapKV

SnapKV 的核心直觉是：不是所有历史 token 对后续生成都同样重要，保留 attention
sink、近期窗口和高重要性历史 token，往往可以在较小 KV cache 下维持可接受的生成
质量。

PyramidSinkKV 在此基础上加入逐层预算分配。`uniform` 模式让每一层使用相同保留
比例；`pyramid` 模式给低层更多预算、高层更少预算；`reversed` 模式作为反向消融；
`spindle` 模式让中间层保留更多、两端层保留更少；`hourglass` 模式让两端层保留更多、
中间层保留更少。所有模式的平均 keep ratio 都近似等于 `--compression_ratio`，
便于公平比较。

## 实现细节

核心代码位于 `pyramidsinkkv.py`。GPTNeoX/Pythia 的 KV cache 预期形状为：

```text
[batch, num_heads, seq_len, head_dim]
```

每层预算由 `--budget_mode {uniform,pyramid,reversed,spindle,hourglass}` 和
`--compression_ratio` 决定。token 保留策略由以下参数控制：

- `--sink_size`
- `--recent_size`
- `--score_method {attention,key_norm,random}`

当前实现只在 prefill 后压缩一次。这样做牺牲了一部分极限压缩潜力，但逻辑更透明，
也更适合课程项目复现。

## RoPE / position_ids 注意事项

GPTNeoX/Pythia 使用 RoPE。KV cache 被压缩后，缓存长度不再等于真实序列长度。如果
直接让 HuggingFace 根据 `past_key_values.get_seq_length()` 推断下一个位置，RoPE
位置会被错误地“重置”到压缩后的长度。

因此，本项目的生成循环维护 `logical_seq_len`。例如 prefill 长度是 1024，压缩后
cache 长度可能是 512，但下一个生成 token 仍然必须使用：

```text
position_ids = 1024
cache_position = 1024
```

代码中有注释说明这一点，`tests/test_pyramidsinkkv.py` 也包含一个小型 sanity check，
验证压缩后的 cache 长度可以不同于逻辑序列长度。

## 实验设置

推荐模型：

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-70m-deduped`

如果需要使用 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

生成实验建议使用较长输出，例如 `--max_new_tokens 256` 或 `512`。Pythia-70M 较小，
输出太短时 TTFT 和 TPOT 的差异容易被计时噪声掩盖。

## Ablation 表格

运行：

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

本次实验在本机 `pyramidsinkkv` conda 环境中运行，模型使用本地缓存的
`EleutherAI/pythia-70m`，设备为 CPU，数值类型为 float32。PPL 为 WikiText-2 raw
test 的小样本 continuation PPL，只使用 1 个 window。random baseline 的 PPL
报告 seeds `0,1,2,3,4` 的均值和标准差；所有生成速度指标报告 3 次重复计时的均值
和标准差。因此下表比单次实验更稳定，但仍不应解读为完整 WikiText 测评。

| Method | Budget mode | Score method | Sink size | Recent size | Compression ratio | PPL | TTFT | TPOT | Throughput | KV memory | Notes |
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

## 结果分析

从这次小样本实验看，no-cache baseline 不保存 KV cache，因此 KV memory 为 0，
但 TPOT 达到 0.0701 s/token，吞吐只有 14.27 tok/s；相比之下 dense KV cache 的
TPOT 为 0.0045 s/token，吞吐为 201.70 tok/s。这说明 KV cache 本身是现代自回归
推理的关键优化，不应该把 dense baseline 理解为“不用 KV”。

所有压缩方法都把 KV memory 从 dense 的 16.50 MB 降到约 9.7 MB，TPOT 维持在约
0.0041-0.0044 s/token，说明压缩后的 cache 确实减少了解码阶段的计算和访存开销，
同时保持了接近 dense KV cache 的速度优势。

质量方面，dense PPL 为 36.88；纺锤形 random 在本次小样本中表现最好，5-seed
平均 PPL 为 45.63，明显优于 uniform random 的 56.31 和 pyramid random 的 65.62。
这提示 Pythia-70M 在该样本上可能更依赖中间层的历史 KV 保留。相反，沙漏型
hourglass 的 PPL 较高，random hourglass 为 77.33，key_norm hourglass 为 85.71，
说明简单地把预算偏向两端层并不适合这个设置。

随机选择在这个小样本上仍然普遍优于 key_norm；例如 spindle random 为 45.63，
spindle key_norm 为 54.93。这个现象不应过度解读，因为本次 PPL 只评估了一个窗口，
但它提示 key_norm 作为简单启发式并不一定稳定优于随机基线。no-sink ablation 的 PPL
上升到 107.09，是本次实验中最差的结果，说明 attention sink 对维持上下文质量
很重要。no-recent-window 的 PPL 为 78.45，也明显差于 random 压缩方法，说明
近期窗口同样有贡献。

## 性能瓶颈讨论

当前实现优先保证正确性和可读性。压缩阶段使用 PyTorch gather，生成阶段使用 Python
循环，因此不是极限优化版本。对于 Pythia-70M，模型很小，Python 调度和 CPU/GPU
同步开销可能占比较高。更大的模型或更长的生成长度更容易体现 KV cache 缩短后的
TPOT 收益。

## 失败案例或局限性

- 只在 prefill 后压缩一次，长时间生成时 cache 仍会继续增长；
- `attention` 打分依赖 attention weights，某些高性能 attention backend 不返回权重；
- 当前 PPL 实现是透明的 continuation PPL，不是高度优化的批量评测；
- 压缩 token 后，attention mask 中被保留 token 的真实绝对位置没有显式存储，当前
  自回归场景下通过绝对 RoPE position 保证新 token 位置正确，但这不是通用稀疏
  attention 框架。

## 结论

PyramidSinkKV 提供了一个适合课程项目的训练免费高效推理基线。它保留了 SnapKV 的
sink/recent/重要 token 思路，同时加入逐层 pyramid 预算分配，并特别处理了
GPTNeoX/Pythia 中 RoPE position_ids 与压缩 cache 长度不一致的问题。下一步应运行
完整 ablation，比较 PPL、TTFT、TPOT、吞吐量和 KV cache memory 后再完成最终分析。
