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
- `demo.py`: 推理演示脚本，展示压缩开启后的长文本生成效果。
- `requirements.txt`: 项目依赖清单。

## 实验结果 (Wikitext-2)
我们在 `wikitext-2-raw-v1` 测试集上对 `Pythia-70M` 进行了不同 KV 缓存容量（Max Capacity）的对比实验。

| 配置 (Configuration) | KV 容量 | 压缩率 | 平均 PPL  |
| :--- | :---: | :---: | :---: |
| **Baseline (Full)** | **500** | **0%** | **50.23** |
| SnapKV-512 | 512 | 0% | 50.00 |
| SnapKV-256 | 256 | 48.8% | 322.86 |
| SnapKV-128 | 128 | 74.4% | 456.08 |
| SnapKV-64 | 64 | 87.2% | 475.46 |

### 结果分析
1. **逻辑正确性验证**：当容量设为 512（略大于 Prompt 长度）时，PPL 回归至基准水平（50.00），证明算法实现逻辑正确。
2. **性能拐点**：对于微型模型 `Pythia-70M`，当压缩率超过 50% 后 PPL 上升显著，反映了小参数规模模型对长程上下文连贯性的高度敏感。
3. **显存效益**：理论上，`SnapKV-64` 可在长文本任务中节省超过 85% 的 KV Cache 显存占用。

## 快速上手

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
### 2. 推理演示
```bash
python demo.py
```
### 3.PPL 评测
```bash
python eval_ppl.py
```