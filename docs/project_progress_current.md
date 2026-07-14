# 项目当前进展与方向

更新时间：2026-07-12

本文用于快速恢复项目上下文：当前项目在做什么、各目录负责什么、主要创新点是什么、下一步卡点在哪里。

## 1. 目录与模块作用

### 根目录核心脚本

- `finetune_llama3_ddi.py`：Llama3 LoRA/QLoRA 微调入口。读取构造好的 JSON 数据，按 `prompt_mode` 组装 chat prompt 并训练 adapter。
- `inference_llama3_ddi.py`：Llama3 + LoRA adapter 推理入口。支持不同 prompt mode、OOM 降级重试、输出格式归一化、可选分类指标。
- `README.md`：项目说明和评估模块更新说明。

### `apps/`

实验命令入口层，主要负责把配置文件转成一次完整运行。

- `run_experiment.py`：统一实验入口。
- `run_train.py`：训练入口封装。
- `run_eval.py`：评估入口封装。
- `run_matrix.py`：批量实验矩阵入口。
- `run_llama3_base_explain_baseline.py`：无微调 Llama3 baseline explanation 推理。
- `aggregate_results.py`：实验结果聚合。

### `configs/`

YAML 配置层。

- `configs/datasets/`：数据路径配置。
- `configs/models/`：模型路径和模型相关默认配置。
- `configs/experiments/`：实验配置，例如分类、解释、无 LoRA baseline、bootstrap 实验。

### `data/`

原始数据、处理中间数据和标准数据文件目录。

- DDI 标签数据：`mechanism/effect/advise/int/false` 及其 test 版本。
- PrimeKG 相关文件：`kg.csv`、图缓存、药名映射等。
- 标准微调/测试 JSON：旧版多为 `queries` 字段，新版正在迁移到 `query_group`。

### `data_process/`

数据构造和蒸馏数据生成层。

- `build_prime_kg.py`：把 PrimeKG CSV 构造成可查询图。
- `build_finetune_dataset.py`：核心数据构造脚本。负责读取 DDI 样本、解析药物 pair、查询 KG、生成 `input`。
- `generate_outputs_distillation.py`：调用外部 LLM 为输入数据生成蒸馏 output。
- `generate_outputs.py`：旧版/本地模型蒸馏生成脚本。
- `extract_train_labels.py` / `extract_test_labels.py`：提取标签 sidecar。
- `data_augmentation.py`：数据增强相关脚本。

### `prime_kg_utils/`

PrimeKG 查询工具层。

- `select.py`：轻量图查询封装，主流程中常用。
- `path_query.py`：多跳路径、树展开、子图可视化等高级探索工具。目前更多是探索工具，尚未深度接入主训练链。

### `pipelines/`

实验编排层。

- `experiment_pipeline.py`：完整实验流程总控。
- `training_pipeline.py`：训练阶段封装。
- `inference_pipeline.py`：推理阶段封装。
- `explanation_pipeline.py`：解释评估阶段封装。
- `classification_pipeline.py`：分类评估封装。
- `compare_pipeline.py`：实验对比。
- `matrix_pipeline.py`：多实验矩阵。
- `results_validator.py`：检查实验产物是否齐全。
- `query_utils.py`：`query_group` 兼容工具，负责把 `- A -> {B, C}` 展开为标准 pair。
- `service_runner.py`：子进程运行工具。

### `evaluate/`

评估模块。

- `classification/`：分类任务评估。
- `explanation/faithfulness.py`：解释事实度评估。
- `explanation/processor.py`：解释输出解析、claim 抽取、query coverage 等。
- `judge/`：LLM-as-judge 评估，目前提示词已收紧，但还需要继续优化。
- `cli/workflows.py`：评估工作流入口，支持 predictions 文件、label 回填、faithfulness 和 judge。

### `modules/`

早期通用封装层，包括数据、模型、训练工具。当前主流程更多走根目录脚本和 `pipelines/`，这里偏历史/辅助模块。

### `regen/`

新数据生成的临时落盘目录。由于 `data/` 里部分文件曾被占用，新版 `query_group` 数据先写到了这里。

### `results/`

实验输出目录。

- `results/test_predictions/`：推理输出。
- `results/eval_report/`：评估报告、faithfulness summary/detail、judge 输出等。

### `tmp_eval_out/`

临时评估输出，用于调试新版 faithfulness 逻辑。

## 2. 当前主要思路

项目目标是面向 DDI 任务构建一条链路：

```text
原始 DDI 数据 + PrimeKG
-> 构造结构化 input
-> 外部 LLM 蒸馏 explanation/reasoning output
-> Llama3 LoRA 微调
-> 推理
-> 分类评估 + 解释事实度评估 + LLM-as-judge
```

当前重点已经从“让模型能分类”推进到“让模型生成更可信、更机制化的解释”。

核心输入结构正在迁移为：

```json
{
  "sentence": "...",
  "query_group": "- A -> {B, C}",
  "kg_evidence": "Mechanism focus: ...\n\nReliable KG anchors:\n...\n\nEvidence summary:\n..."
}
```

其中：

- `sentence`：临床原句。
- `query_group`：需要分析的药物 pair，替代旧字段 `queries`。
- `kg_evidence`：精简后的 KG 机制证据，不再重复 query 信息。

## 3. 当前创新点

### 3.1 KG 不再作为节点堆叠，而是机制证据摘要

原始 PrimeKG 很大，直接塞上下文会带来大量噪声。当前策略是限制在一阶邻居，但不再简单输出所有邻居，而是做：

- 可靠 KG anchor 过滤。
- 去除低质量 PPI / 泛化 Gene-protein 噪声。
- 保留更有 DDI 解释价值的 `enzyme / transporter / target`。
- 将 KG evidence 压缩成模型更容易利用的机制上下文。

### 3.2 `query_group` 替代 `queries`

旧版逐 pair 枚举会重复、冗长。新版使用 grouped query：

```text
- desglymidodrine -> {metformin, cimetidine, ranitidine}
```

好处：

- 减少 prompt token。
- 更贴近“一个句子描述一组共机制 pair”的真实场景。
- 便于后续 explanation 输出做 phenomenon-level reasoning。

### 3.3 句子条件 KG evidence

KG 节点是否有用，不只看药物本身，还看当前句子 cue。

例如：

- 句子提到 `metabolism / inhibit / concentration`，优先 enzyme，尤其 `CYP / UGT`。
- 句子提到 `renal / secretion / clearance`，优先 transporter，尤其 `SLC22 / SLC47`。
- 句子提到 `QT / potassium / receptor / antagonize`，优先 target 或 ion channel。

### 3.4 机制槽位映射

正在设计的规则表：

- `ANCHOR_PREFIX_RULES`：把 `CYP / UGT / SLC22 / SLC47 / SLCO / ABC / KCN / SCN / ADRA / HRH` 等 anchor 映射到机制槽位。
- `FAMILY_MECHANISM_PHRASES`：把 anchor family 转成稳定机制短语。

目标是规则生成：

```text
Evidence summary:
- Dominant mechanism slot: enzyme.
- Observed anchor families: CYP, UGT, ABCB, SLCO.
- Strong evidence: ketoconazole has CYP/UGT metabolism anchors aligned with the sentence-level metabolism cue.
- Weak evidence: no reliable mechanism-level KG anchor is found for VIRACEPT.
```

### 3.5 事实度评估更严格

评估模块已完成关键修复：

- `predicted_label` 可从第一阶段 label-only 结果回填。
- 不再把 JSON 字符串符号当成自然语言 claim。
- `representative_pairs` 不再计入 hallucination claims。
- 区分 `supported / partial / unsupported`。
- 对句外医学推断更谨慎，避免把“医学上合理”误判为“当前证据支持”。

## 4. 当前已完成的重要改动

- 新增 `query_group` 兼容工具：`pipelines/query_utils.py`。
- 数据构造、蒸馏、微调、推理、评估均已支持优先读取 `query_group`，并兼容旧 `queries`。
- `build_finetune_dataset.py` 已能生成新版 `query_group` input。
- `kg_evidence` 已去掉内嵌 `Query group`。
- 新版数据已生成到 `regen/`，因为 `data/` 中部分旧文件被外部进程占用。
- 无 LoRA Llama3 baseline 已跑过一版 no-KG explanation，并用新版 faithfulness 评估。
- README 已记录评估模块更新。

## 5. 当前主要卡点

### 5.1 KG anchor 质量不稳定

问题：

- 部分药物在 PrimeKG 中命中质量差。
- 一阶邻居可能全是无关 PPI。
- 有些药物 pair 没有共享节点，但句子明显是 Mechanism / Effect。

可能解决思路：

- 增强 reliable anchor 过滤。
- 明确丢弃低质量 `ppi` 节点。
- 对无可靠 anchor 的药物诚实输出 `no reliable mechanism-level KG anchor found`。
- 对非 False 且无共享节点的样本，用机制槽位和句子 cue 做弱桥接，而不是硬造共享节点。

### 5.2 `Evidence summary` 规则还未正式落代码

问题：

- 当前 `kg_evidence` 仍有 `Pair-level bridge evidence` 和 `Interpretation hint` 风格。
- 新目标是替换为更规则化的 `Evidence summary`。

计划：

- 新增 `ANCHOR_PREFIX_RULES`。
- 新增 `FAMILY_MECHANISM_PHRASES`。
- 用规则生成：
  - `Dominant mechanism slot`
  - `Observed / Repeated anchor families`
  - `Strong evidence`
  - `Weak evidence`
- 去掉 `Missing evidence` 独立段，把证据不足统一放入 `Weak evidence`。

### 5.3 PrimeKG 关系语义不足

问题：

- PrimeKG 往往只说明药物与某个酶/转运体/靶点有关。
- 它不一定说明是底物、抑制剂、诱导剂或具体方向。

可能解决思路：

- KG input 只做机制锚点，不直接给结论。
- 外部 LLM 蒸馏时负责补充机制推理链。
- 评估时区分：
  - KG anchor 支持。
  - 句子支持。
  - 外部医学推断。

### 5.4 LLM-as-judge 还需要继续优化

问题：

- 早期 judge 容易把“医学上合理”当成“由当前证据支持”。
- 当前 prompt 已收紧，但还没有系统全量验证。

可能解决思路：

- judge 输入中加入 deterministic faithfulness diagnostics。
- 打分维度区分：
  - 机制链完整性。
  - 方向正确性。
  - 证据支持性。
  - 不确定性校准。
- 明确要求 judge 不用外部知识补全缺失证据。

### 5.5 数据文件落盘和版本管理

问题：

- `data/` 下旧文件曾被进程占用，导致无法覆盖。
- 新版数据暂时落在 `regen/`。

可能解决思路：

- 后续实验配置显式指向 `regen/` 新版数据。
- 稳定后再替换 `data/` 标准文件。
- 数据文件名加入 schema 标识，如 `query_group_with_kg`，降低混淆。

## 6. 下一步建议

优先级从高到低：

1. 在 `build_finetune_dataset.py` 中正式落地 `Evidence summary` 规则。
2. 重新生成 with-KG 训练集和测试集。
3. 用外部 LLM 对新版 with-KG input 做蒸馏输出。
4. 微调 Llama3 with-KG explanation / reasoning 版本。
5. 用新版 faithfulness 评估 with-KG 和 no-KG 的差异。
6. 单独优化并验证 LLM-as-judge。

## 7. 一句话总结

当前项目的核心方向是：把 PrimeKG 从“大图节点列表”压缩成“句子条件下的机制证据摘要”，再用外部 LLM 蒸馏出高质量推理链，最后用 Llama3 微调学习这种机制解释能力，并用更严格的事实度评估控制幻觉。
