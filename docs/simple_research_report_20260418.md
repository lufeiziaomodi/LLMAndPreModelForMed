# DDI关系识别与可解释推理流程简报（更新于2026-04-19）

## 1. 研究目标
本项目面向药物-药物相互作用（DDI）识别，构建了一个从“蒸馏数据生成 -> LoRA微调 -> 推理 -> 解释评估/LLM评审”的统一实验链路。当前重点为：

- 第一层：分类标签识别能力（Mechanism/Effect/Advice/Int/False）。
- 第二层：可解释推理生成能力（现象级解释，支持评估）。
- 第三层：解释质量评估（规则化 faithfulness + LLM-as-judge）。

## 2. 任务定义与标签体系

### 2.1 标签体系（5类）
- Mechanism：偏药代机制（如代谢、吸收、转运）。
- Effect：偏药效/临床效应变化。
- Advice：监测、禁忌、给药建议。
- Int：确认存在交互，但机制/效应细节不足。
- False：指定药物对不存在交互。

### 2.2 解释输出范式（当前主线）
在 `explanation_without_kg` 模式下，输出要求为 JSON 列表，核心字段：
- query（固定为 GLOBAL_PHENOMENON）
- analysis_steps（简短编号步骤）
- mechanism_summary（机制概述）
- representative_pairs（最多3对代表性药物对）

该设计的目的：减少“逐query全枚举”导致的长链噪声和虚构扩写。

## 3. 全流程（数据 -> 训练 -> 推理 -> 评估）

### 3.1 蒸馏数据构建（无KG解释蒸馏）
脚本：`data_process/generate_outputs_distillation.py`

核心做法：
- 使用 OpenAI-compatible 接口调用 `qwen-plus` 生成训练输出。
- generation_mode 使用 `distill_reasoning_without_kg`。
- 强制 `ignore_kg_evidence=true`，只基于 sentence 生成解释。
- few-shot 提示词中明确“GLOBAL_PHENOMENON + representative_pairs<=3”。

输出文件：
- `data/finetune_dataset_input_reasoning_without_kg.json`

### 3.2 LoRA微调
脚本：`finetune_llama3_ddi.py`

核心做法：
- 基座模型：`models/Meta-Llama-3-8B-Instruct`
- QLoRA（默认4bit），LoRA目标层：
  - `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 数据通过 `tokenizer.apply_chat_template` 构造成 chat SFT 样本。
- Prompt 模式：`explanation_without_kg`（与蒸馏输出一致）。

输出目录：
- `results/llama3_ddi_lora_reasoning_without_kg`

### 3.3 推理
脚本：inference_llama3_ddi.py

关键点：
- 推理提示词与训练保持一致（explanation_without_kg）
- 解释后处理采用容错解析：
  1) 解析近似JSON
  2) 归一化到 GLOBAL_PHENOMENON 结构
  3) representative_pairs 缺失时，从 query 回填（最多3个）

### 3.4 统一编排
入口：python -m apps.run_experiment --config configs/experiments/restart_explanation_bootstrap.yaml

编排层：pipelines/experiment_pipeline.py
- training -> inference -> explanation_eval(+judge)
- 输出统一到 output_root/experiment_name

## 4. 评估模块（代码口径）

本节是本次更新重点。相比旧版文档，当前代码已经从“单一阈值匹配”升级为“结构化claim + 严格/宽松双视角 + 0-2打分rubric Judge”。

### 4.1 Faithfulness：核心方法
主文件：evaluate/explanation/faithfulness.py
辅助处理：evaluate/explanation/processor.py

处理步骤：
1. 解释归一化
- normalize_explanation_output 会统一解析：
  - 结构化dict格式
  - 旧版list格式
  - 原始文本fallback
- 若 predicted_label 缺失，尝试从解释文本回填标签（label_backfilled）。

2. claim抽取
- extract_claim_records 优先使用 core_claims。
- 若缺失，退化到 mechanism_summary / analysis_steps / plain_text。
- 同时单独保留 optional_inference（可选推断性陈述）。

3. 支持度特征
- claim 与 evidence（sentence + 可选kg）计算：
  - jaccard
  - token_precision
  - overlap_count
  - exact_substring
  - unsupported_tokens / novel_unsupported_tokens

4. 支持类别判定（不再是固定support_threshold=0.2）
- _categorize_support 返回三类：supported / partial / unsupported
- 同时给出两个视角：
  - strict_sentence_grounding（严格句证）
  - lenient_medical_plausibility（宽松医学合理性）

5. 查询覆盖率
- compute_query_coverage 比较 query_pairs 与 representative_pairs 的覆盖。
- 输出 covered_queries / total_queries / query_coverage。

6. 主要输出字段
- coverage_ratio
- partial_support_rate
- hallucination_rate
- consistency_score
- query_coverage
- strict_sentence_grounding
- lenient_medical_plausibility
- hallucination_types
- supported_claims / partial_supported_claims / unsupported_claims
- optional_inference_claims / unsupported_optional_inference_claims

### 4.2 Faithfulness：聚合指标
aggregate_faithfulness 当前汇总字段包含：
- coverage_mean
- hallucination_mean
- partial_support_mean
- consistency_mean
- query_coverage_mean
- strict_supported_mean
- strict_unsupported_mean
- lenient_supported_mean
- kg_grounded_mean（仅with_kg可用）
- by_evidence_mode（with_kg / no_kg分桶）

### 4.3 LLM-as-Judge：核心方法
提示词：evaluate/judge/judge_prompts.py
执行器：evaluate/judge/qwen_judge.py
结果归一化：evaluate/judge/result_normalizer.py

当前Judge不是旧版1-5维度直评，而是先按0-2 rubric打分，再映射兼容字段。

Judge输入包括：
- sentence / queries / kg_evidence / predicted_label / explanation
- 来自Faithfulness的确定性诊断：
  - supported_claims
  - partial_supported_claims
  - unsupported_claims
  - query_coverage诊断

Rubric维度（0-2）：
- label_alignment
- sentence_grounding
- phenomenon_completeness
- entity_pair_coverage
- non_hallucination
- clarity

Rubric总分：
- overall_score，范围0-12
- overall_decision：poor / fair / good

Judge 现在只输出机制向 rubric，不再做旧版 faithfulness / label_alignment 兼容映射。

### 4.4 Judge汇总字段（最新版）
run_judge 只汇总机制向字段：

- mechanism_overall_score_mean
- mechanism_chain_completeness_mean
- mechanism_direction_correctness_mean
- mechanism_granularity_mean
- mechanism_internal_consistency_mean
- uncertainty_calibration_mean
- clinical_actionability_mean
- mechanism_good_rate

## 5. 核心参数清单（当前配置）

配置文件：configs/experiments/restart_explanation_bootstrap.yaml

### 4.1 蒸馏参数
- model: `qwen-plus`
- generation_mode: `distill_reasoning_without_kg`
- ignore_kg_evidence: `true`
- max_new_tokens: `768`
- max_workers: `4`
- save_every: `100`
- overwrite: `true`

### 4.2 微调参数
- prompt_mode: `explanation_without_kg`
- batch_size: `1`
- grad_accum: `8`
- learning_rate: `2e-4`
- max_steps: `1000`
- lora_r: `16`
- lora_alpha: `32`
- train_test_split: `0.05`
- multi_gpu_shard: 启用

### 5.3 推理
- prompt_mode: explanation_without_kg
- temperature: 0.0
- top_p: 1.0
- max_new_tokens: -1（脚本内部按模式回落默认）
- multi_gpu_shard: enabled

### 5.4 评估与Judge
EvaluationConfig关键项（evaluate/config.py）：
- predictions_file
- predictions_labels_file
- label_predictions_file
- output_dir
- use_judge
- judge_model_id
- judge_api_key
- judge_max_retries
- judge_retry_delay
- judge_checkpoint_every

## 6. 当前结果解读（与代码口径一致）

目前目录中可见的历史结果显示：
- Faithfulness 仍保留严格证据约束，用于判断解释是否过度外推。
- Judge 已改成纯机制导向，用于判断解释是否真正讲清机制链条。

两者现在是并列关系，不再互相引用对方的中间产物。

## 7. 结论与建议

1. 文档与代码口径已对齐：当前评估模块是“faithfulness 负责底线，Judge 负责机制”的双层结构。
2. 后续汇报建议优先展示 Judge 的机制字段作为主结果，faithfulness 作为约束背景。
3. 若需要强化第一层分类结论，建议单独运行 label_only 路径并产出独立分类报告。

## 8. 机制导向评估改造（写入项目文档）

针对“解释链条目标是机制挖掘，而非停留标签表面”，建议在现有评估上做以下口径升级：

1. 保留 faithfulness 的事实约束层，但不让它进入 Judge。
2. 提升机制深度层为主指标，重点考核机制链完整性、方向正确性、粒度与一致性。
3. 增加临床可用层，考核风险与干预建议的可执行性。

推荐总分权重：

- 事实约束层 20%
- 机制深度层 50%
- 机制方向与一致性 20%
- 临床可用层 10%

落地说明：

- 现有 faithfulness 保持不变，只负责支持/部分支持/不支持的证据诊断。
- Judge 改为机制 rubric（0-2），不再读取 faithfulness 中间结果。
- 汇报时将“可靠性表”和“机制能力表”分开展示。

详细方案见：

- [docs/mechanism_evaluation_framework_zh.md](../docs/mechanism_evaluation_framework_zh.md)
