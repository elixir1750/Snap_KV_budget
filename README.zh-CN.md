# Pythia-70M KV 缓存预算

本仓库研究**无训练**的推理时 KV 缓存预算与压缩方法，针对  
`EleutherAI/pythia-70m` 和 `EleutherAI/pythia-70m-deduped` 模型。  
骨干模型权重**从不更新**，所有改动都在推理时进行，通过改变预填充后 GPTNeoX/Pythia 的 `past_key_values` 保留方式实现。

项目包含：

- 无 KV 压缩的密集生成基线
- 使用 `use_cache=False` 的无缓存基线
- 随机 KV 基线
- 均匀、金字塔、反向金字塔、纺锤形、沙漏形层间 KV 预算
- 基于层预算的 KV 压缩（随机或 SnapKV token 选择）
- 反向金字塔、纺锤形、沙漏形、无 sink token、无最近窗口等消融实验
- 生成速度基准、困惑度评估、消融表格生成、终端流式演示

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

代码直接使用 HuggingFace API，镜像设置通过标准环境变量处理，例如：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后加载以下模型之一：

```bash
EleutherAI/pythia-70m
EleutherAI/pythia-70m-deduped
```

## 方法

`pyramidsinkkv.py` 包含核心实现。对每一层：

1. 计算保留预算
2. 始终保留 sink token 和最近 token
3. 从中间 token 中按策略选择剩余部分
4. 按原始顺序收集 KV 张量

预期的 GPTNeoX/Pythia 缓存张量形状：

```text
[batch, num_heads, seq_len, head_dim]
```

预算模式：

- `uniform`：每层保留相同比例
- `pyramid`：低层保留更多，高层保留更少
- `reversed`：低层保留更少，高层保留更多
- `spindle`：中间层保留更多，边缘层保留更少
- `hourglass`：边缘层保留更多，中间层保留更少
- `dense`：禁用压缩（用于评估脚本）
- `no_cache`：完全禁用 KV 缓存，每步重新计算完整前缀

`budget_mode` 和 `score_method` 分别解决两个不同的问题：  
`budget_mode` 决定每层保留多少 token，`score_method` 决定在预留 sink 和最近 token 后保留哪些中间 token。

选择方法：

- `random`：可复现的随机中间 token 选择（使用 `--seed`）
- `key_norm`：仅内部回退，选择键向量 L2 范数最大的中间 token（注意力权重不可用时使用）
- `attention`：根据最近查询的注意力分数对历史 token 评分；若注意力权重不可用则记录警告并回退到 `key_norm`
- `snapkv`：**推荐**的基于注意力的方法。预填充时读取注意力权重，取最后 `--observation_window` 个查询位置，先在 batch 和查询位置上平均，再用 `--snapkv_head_aggregation mean|max|per_head` 处理不同 head 的分数，每层选择得分最高的中间 token。Sink token 和最近 token 仍被始终保留。默认 `mean` 保持原行为；`max` 表示只要某个 head 强烈关注某个 token，就把它视为重要 token，但所有 head 仍共享同一组索引；`per_head` 最接近完整 SnapKV，每个 attention head 会独立选择并 gather 自己的 token。`--snapkv_pooling_kernel 3/5` 可在 top-k 前对 token 分数做长度保持的一维平均池化。

SnapKV 需要注意力权重。加载器会尽可能通过 `attn_implementation="eager"` 请求 eager attention，若遇到旧版本则重试。如果注意力权重仍然不可用，结果会记录 `score_method_fallback=true` 并使用 `key_norm`，不会静默改变行为。返回注意力权重可能增加首 token 时间；本实现优先考虑正确性和可复现性，而非融合内核的速度。

## RoPE 正确性

GPTNeoX/Pythia 使用 RoPE，因此压缩后的缓存长度不能直接当作真实 token 位置。手动生成循环维护了 `logical_seq_len` 计数器。预填充后 KV 缓存可能从 1024 token 压缩到 512 token，但下一个生成 token 仍然接收 `position_ids=1024` 和 `cache_position=1024`。  
因此脚本使用 `pyramidsinkkv.generate(...)` 而非默认的 `model.generate(...)`。`tests/test_pyramidsinkkv.py` 中的冒烟测试也检查压缩后缓存长度与逻辑序列长度是否可以不同。

## 运行实验

`scripts.eval_ppl` 用于 continuation PPL，`scripts.benchmark_generation`
用于 TTFT/TPOT/throughput，`scripts.run_ablation` 用于生成分组表格。三个入口共享大部分核心参数：

- 将 `--budget_mode` 替换为 `dense`、`no_cache`、`uniform`、`pyramid`、`reversed`、`spindle` 或 `hourglass`。
- 将 `--score_method` 替换为 `random` 或 `snapkv`。`key_norm` 主要作为 attention 不可用时的内部 fallback。
- SnapKV 可比较 `--snapkv_head_aggregation mean`、`max`、`per_head`。其中 `per_head` 最接近完整 SnapKV，因为每个 head 会 gather 自己的索引。
- `--snapkv_pooling_kernel 1` 表示不做 pooling，`3/5` 用于测试局部平滑。
- `--compression_ratio 0.75/0.5/0.25` 可用于保留比例 sweep。
- `--dataset wikitext` 适合快速缓存实验；长文本可用 `--dataset redpajama --redpajama_source hub --redpajama_hub_config wikipedia`。RedPajama loader 会先尝试原始 JSONL URL，如果流式连接不稳定，会自动回退到 HuggingFace converted parquet 分片。
- GPU 使用 `--device cuda --dtype float16`；CPU 复现实验可用 `--device cpu --dtype float32`。
- random baseline 建议跑 `0,1,2,3,4` 多个 seed 并报告均值和标准差；固定窗口下 SnapKV 是确定性的。

主要指标包括 `ppl`、`TTFT`、`TPOT`、`throughput`、`total_time`、`kv_cache_memory_mb`、`achieved_compression_ratio`、`scoring_overhead_sec`、`snapkv_fallback_status` 和 `notes`。如果实验失败，ablation runner 会在备注中记录错误，不会编造结果。

下面是一个 RedPajama PPL 示例：使用 spindle 预算、完整 per-head SnapKV、pooling 和 25% 保留比例。

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

如果要测生成速度，保留同样的方法参数并把入口换成 `scripts.benchmark_generation`；如果要跑分组实验，使用 `scripts.run_ablation` 并设置 `--score_methods random,snapkv` 和需要的逗号分隔 `--budget_modes`。

## 终端速度演示

```bash
python -m scripts.demo_speed_animation \
  --model_name_or_path EleutherAI/pythia-70m \
  --methods dense,pyramid \
  --max_new_tokens 256 \
  --compression_ratio 0.5 \
  --sink_size 4 \
  --recent_size 64
```

若安装了 `rich`，演示会显示动态面板；否则回退到纯标准输出刷新。

## 冒烟测试

```bash
python -m pytest tests
```

或不使用 pytest：

```bash
python tests/test_pyramidsinkkv.py
python -m compileall pyramidsinkkv.py scripts tests
```

## 说明

- 本方法无需训练：没有可训练参数、没有微调、没有模型权重更新。
- 初始实现仅在预填充后压缩一次。解码过程中重复压缩被有意排除，以保证课程项目基线的可靠性和易检查性。
- 基于注意力的选择需要注意力权重。若当前注意力实现不返回注意力权重，代码会回退到 key-norm 评分，记录警告并在 JSON/CSV/Markdown 输出中记录回退状态。
