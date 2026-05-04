# SnapKV-Pythia: 语言模型高效推理 (逐层 KV Cache 压缩)

本项目为 **上海交通大学 NLP 课程大作业** 的个人部分实现。我们针对 `Pythia-70M` 模型实现了 **SnapKV** 算法，通过逐层 KV Cache 压缩技术，在不改变模型参数且无需微调的情况下，探索推理效率与模型性能之间的权衡。

## 项目亮点
- **算法实现**：完整复现了 SnapKV 核心逻辑，包含观察窗口（Observation Window）、基于池化的注意力热点选择（Heavy Hitter Selection）及起始词保护（Attention Sinks）。
- **通用适配**：通过包装器（Wrapper）模式注入模型补丁，完美兼容 HuggingFace `transformers` 库的最新 `DynamicCache` 结构。
- **环境友好**：支持纯 CPU 推理，代码经过优化，可在普通笔记本上完成全量评测。

## 文件组织
- `snapkv_utils.py`: 核心算法库，包含注意力分数平滑处理与 Top-K 索引选择逻辑。
- `modify_gptneox.py`: 模型注入补丁，通过实例拦截技术向 `GPTNeoX` 架构注入压缩逻辑。
- `eval_ppl.py`: 自动化评测脚本，支持在 `Wikitext-2` 数据集上进行不同容量的 PPL 对比测试。
- `benchmark_speed.py`: 速度与加速比评测脚本，支持 TTFT、TPOT 和 Throughput 测试。
- `demo.py`: 推理演示脚本，展示压缩开启后的长文本生成效果。
- `requirements.txt`: 项目依赖清单。

## 实验结果 (Wikitext-2)

### 1. 精度评测 (PPL)
我们在 `wikitext-2-raw-v1` 测试集上对 `Pythia-70M` 进行了不同 KV 缓存容量（Max Capacity）的对比实验（Prompt 长度 $\approx$ 500 tokens）。

| 配置 (Configuration) | KV 容量 | 压缩率 | 平均 PPL |
| :--- | :---: | :---: | :---: |
| **Baseline (Full)** | **500** | **0%** | **50.23** |
| SnapKV-512 | 512 | 0% | 50.00 |
| SnapKV-256 | 256 | 48.8% | 322.86 |
| SnapKV-128 | 128 | 74.4% | 456.08 |
| SnapKV-64 | 64 | 87.2% | 475.46 |

### 2. 加速与吞吐量评测
我们在 CPU 环境下对长文本生成任务（Prompt 长度 $\approx$ 1000 tokens，生成 50 tokens）进行了效率测试。SnapKV 设定容量为 64。

| 指标 | Baseline | SnapKV | 提升比 |
| :--- | :---: | :---: | :---: |
| **TTFT** (Time To First Token, 首字延迟) | 0.8259 s | 0.7233 s | **1.14x** |
| **TPOT** (Time Per Output Token, 逐字延迟) | 0.0134 s | 0.0082 s | **1.65x** |
| **Throughput** (吞吐量, tokens/s) | 33.6947 | 44.5314 | **1.32x** |

## 结果与效益分析
1. **逻辑正确性验证**：当容量设为 512（略大于 Prompt 长度）时，PPL 回归至基准水平（50.00），证明算法实现逻辑正确。
2. **性能与连贯性权衡**：对于微型模型 `Pythia-70M`，当压缩率超过 50% 后 PPL 上升显著，反映了小参数规模模型对长程上下文连贯性的高度敏感。
3. **显著的生成提速**：在生成阶段（Decoding），由于 KV Cache 被恒定压缩在 64，无需扫描全量长文本，单字生成延迟（TPOT）大幅缩短，实现了 **1.65 倍** 的推理加速。
4. **显存效益**：理论上，`SnapKV-64` 在长文本任务中可使 KV Cache 显存占用**降低超过 85%**，这为大模型在显存严重受限的设备上部署提供了可能。

## 快速上手

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
### 2. 推理演示
```bash
python demo.py
```
### 3. PPL 评测
```bash
python eval_ppl.py
```
### 4. 速度基准测试
```bash
python benchmark_speed.py
```