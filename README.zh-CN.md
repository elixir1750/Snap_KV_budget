# Pythia-70M KV 缓存预算

本仓库研究**无训练**的推理时 KV 缓存预算与压缩方法，针对  
`EleutherAI/pythia-70m` 和 `EleutherAI/pythia-70m-deduped` 模型。  
骨干模型权重**从不更新**，所有改动都在推理时进行，通过改变预填充后 GPTNeoX/Pythia 的 `past_key_values` 保留方式实现。

## 包含内容

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

### 预算模式

- `uniform`：每层保留相同比例
- `pyramid`：低层保留更多，高层保留更少
- `reversed`：低层保留更少，高层保留更多
- `spindle`：中间层保留更多，边缘层保留更少
- `hourglass`：边缘层保留更多，中间层保留更少
- `dense`：禁用压缩（用于评估脚本）
- `no_cache`：完全禁用 KV 缓存，每步重新计算完整前缀

`budget_mode` 决定每层保留多少 token，`score_method` 决定在预留 sink 和最近 token 后保留哪些中间 token。

### 选择方法

- `random`：可复现的随机中间 token 选择（使用 `--seed`）
- `key_norm`：仅内部回退，选择键向量 L2 范数最大的中间 token（注意力权重不可用时使用）
- `attention`：根据最近查询的注意力分数对历史 token 评分；若注意力权重不可用则记录警告并回退到 `key_norm`
- `snapkv`：**推荐**的基于注意力的方法。预填充时读取注意力权重，取最后 `--observation_window` 个查询位置的平均分数，每层选择得分最高的中间 token。Sink token 和最近 token 仍被始终保留。

SnapKV 需要注意力权重。加载器会尽可能通过 `attn_implementation="eager"` 请求 eager attention，若遇到旧版本则重试。如果注意力权重仍然不可用，结果会记录 `score_method_fallback=true` 并使用 `key_norm`，不会静默改变行为。

## RoPE 正确性

GPTNeoX/Pythia 使用 RoPE，因此压缩后的缓存长度不能直接当作真实 token 位置。手动生成循环维护了 `logical_seq_len` 计数器。预填充后 KV 缓存可能从 1024 token 压缩到 512 token，但下一个生成 token 仍然接收 `position_ids=1024` 和 `cache_position=1024`。  
因此脚本使用 `pyramidsinkkv.generate(...)` 而非默认的 `model.generate(...)`。

## 生成基准测试

### 密集基线

```bash
python -m scripts.benchmark_generation \
  --model_name_or_path EleutherAI/pythia-70m \
  --budget_mode dense \
  --score_method random \
  --max_new_tokens 256 \
  --output_json results/dense_generation.json
```

### PyramidSinkKV（金字塔 + 随机选择）

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

### 均匀 SnapKV

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

### 金字塔 SnapKV

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

### 纺锤形 SnapKV

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

### 报告指标

- `TTFT`：首 token 时间（包含预填充和可选的缓存压缩）
- `TPOT`：首个生成 token 后每个输出 token 的平均时间
- `throughput`：每秒生成 token 数
- `total_time`：总生成时间
- `scoring_overhead_sec`：额外的评分开销（当前 SnapKV 使用预填充返回的注意力权重，该成本已计入 TTFT）
- `compression_overhead_sec`：索引选择和 K/V 收集时间
- `kv_cache_memory_mb`：最终 KV 缓存估算内存
- `achieved_compression_ratio`：实际压缩后 token 数 / 预填充后原始 token 数
- `snapkv_fallback_status`：SnapKV 是否使用了真正的注意力选择，还是回退到 `key_norm`

## 困惑度评估

### WikiText

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

### WikiText + 纺锤形 + SnapKV

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

### PG-19 单样本快速实验

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

对于密集模型困惑度，使用 `--budget_mode dense`。

## 消融表格

运行完整的小型消融套件：

```bash
python -m scripts.run_ablation \
  --model_name_or_path EleutherAI/pythia-70m \
  --compression_ratio 0.5 \
  --max_new_tokens 256 \
  --max_windows 2
```

输出文件：

- `results/group_main_ablation.json`
- `results/group_main_ablation.csv`
- `results/group_main_ablation.md`
- `results/group_ratio_sweep.json`
- `results/group_ratio_sweep.csv`
- `results/group_ratio_sweep.md`
- `results/group_sink_recent_ablation.json`
- `results/group_sink_recent_ablation.csv`
- `results/group_sink_recent_ablation.md`

SnapKV 分组消融：

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

生成的 Markdown 表格可直接复制到最终课程报告中。若某个实验失败，运行器会在备注中记录错误，不会编造结果。

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