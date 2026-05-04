# SnapKV-Pythia: 语言模型高效推理 (逐层 KV Cache 压缩)

本项目为 **上海交通大学 NLP 课程大作业** 的个人部分实现。我们针对 `Pythia-70M` 模型实现了 **SnapKV** 算法，通过逐层 KV Cache 压缩技术，在不改变模型参数且无需微调的情况下，探索推理效率与模型性能之间的权衡。

## 项目亮点
- **算法实现**：完整复现了 SnapKV 核心逻辑，包含观察窗口（Observation Window）、基于池化的注意力热点选择（Heavy Hitter Selection）及起始词保护（Attention Sinks）。
- **底层机制修复（RoPE 对齐）**：深度分析并修复了 KV Cache 长度截断后导致的 HuggingFace 底层 RoPE（旋转位置编码）Position ID 错位问题，成功挽救了压缩后的性能崩塌。
- **通用适配**：通过包装器（Wrapper）模式注入模型补丁，完美兼容 HuggingFace `transformers` 库的最新 `DynamicCache` 结构。
- **环境友好**：支持纯 CPU 推理，代码经过优化，可在普通笔记本上完成全量评测。

## 文件组织
- `snapkv_utils.py`: 核心算法库，包含注意力分数平滑处理与 Top-K 索引选择逻辑。
- `modify_gptneox.py`: 模型注入补丁，通过实例拦截技术向 `GPTNeoX` 架构注入压缩逻辑与位置编码对齐逻辑。
- `eval_ppl.py`: 自动化评测脚本，支持在 `Wikitext-2` 数据集上进行不同容量的 PPL 对比测试。
- `benchmark_speed.py`: 速度与加速比评测脚本，支持 TTFT、TPOT 和 Throughput 测试。
- `demo.py`: 推理演示脚本，展示压缩开启后的长文本生成效果。
- `requirements.txt`: 项目依赖清单。

## 实验结果 (Wikitext-2)

### 1. 精度评测 (PPL)
我们在 `wikitext-2-raw-v1` 测试集上对 `Pythia-70M` 进行了不同 KV 缓存容量（Max Capacity）的对比实验（Prompt 长度 $\approx$ 500 tokens）。

| 配置 (Configuration) | KV 容量 | 压缩率 (约) | 平均 PPL |
| :--- | :---: | :---: | :---: |
| **Baseline (Full)** | **~500** | **0%** | **50.23** |
| SnapKV-512 | 512 | 0% | 50.23 |
| SnapKV-256 | 256 | 48.8% | 52.40 |
| SnapKV-128 | 128 | 74.4% | 61.60 |
| SnapKV-64 | 64 | 87.2% | 69.37 |

### 2. 加速与吞吐量评测
我们在 CPU 环境下对长文本生成任务（Prompt 长度 $\approx$ 1000 tokens，生成 50 tokens）进行了效率测试。SnapKV 设定容量为 64。

| 指标 | Baseline | SnapKV | 提升比 |
| :--- | :---: | :---: | :---: |
| **TTFT** (Time To First Token, 首字延迟) | 2.7099 s | 2.3970 s | **1.13x** |
| **TPOT** (Time Per Output Token, 逐字延迟) | 0.0363 s | 0.0242 s | **1.50x** |
| **Throughput** (吞吐量, tokens/s) | 11.1393 | 13.9490 | **1.25x** |

## 结果与效益分析
1. **逻辑正确性验证**：当容量设为 512（略大于 Prompt 长度）时，PPL 与 Baseline 保持完全一致（均为 50.23），证明算法在无损状态下的实现逻辑完全正确，未引入额外计算误差。
2. **极强的压缩鲁棒性（克服 RoPE 错位）**：在修复了 RoPE 位置编码错位问题后，SnapKV 展现出了惊人的效率-性能平衡。即使在压缩率接近 50%（Capacity=256）时，PPL 仅发生微小波动（50.23 $\rightarrow$ 52.40）。这强有力地证明了 SnapKV 提取“注意力热点”的策略能够精准保留关键上下文。
3. **极限压缩下的平滑退化**：即使在保留容量仅为 64（压缩率超 87%）的极端情况下，PPL 也只是平滑退化至 69.37，且模型仍能输出连贯的文本，这大大优越于早期未对齐位置编码时的崩塌现象。
4. **显著的生成提速与显存节省**：在生成阶段（Decoding），由于 KV Cache 被恒定压缩在 64，无需再计算和扫描全量的长文本缓存，单字生成延迟（TPOT）大幅缩短，实现了 **1.50 倍** 的推理加速（Throughput 吞吐量提升了 **1.25 倍**）。同时，KV Cache 的显存开销**降低超过 85%**，极大缓解了长序列推理的显存瓶颈。

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