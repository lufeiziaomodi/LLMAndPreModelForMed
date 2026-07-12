# 机制导向可解释评估框架（中文）

## 1. 目标

本框架用于支持“机制挖掘优先”的可解释评估，不再把标签一致性作为唯一核心目标。

核心思想：

- 标签正确只是必要条件，不是充分条件。
- 解释是否揭示“药物对 -> 生物过程 -> 结果”的机制链条，才是主目标。

## 2. 三层评估结构

### 2.1 事实约束层（底线）

用途：防止句外编造。

建议指标：

- strict_sentence_grounding（来自 faithfulness）
- hallucination_rate（来自 faithfulness）
- unsupported_claims 占比
- entity/query 对齐（query_coverage）

定位：只做约束门槛，不作为主导总分。

### 2.2 机制深度层（主指标）

用途：衡量机制解释能力。

建议维度：

- mechanism_chain_completeness：是否覆盖“施动药 -> 机制过程 -> 受体药/结果”。
- mechanism_direction_correctness：方向是否正确（谁影响谁，增加还是减少）。
- mechanism_granularity：是否达到可验证机制粒度（酶/转运体/吸收/排泄/受体等）。
- mechanism_internal_consistency：analysis_steps 与 mechanism_summary 是否一致。
- uncertainty_calibration：证据不足时是否明确不确定性，而非强断言。

### 2.3 临床可用层（加分）

用途：连接机制与应用。

建议维度：

- clinical_actionability：是否指出风险类型或给出监测/避免合用/剂量调整建议。

说明：

- 这一层级由 Judge 独立完成，不读取 faithfulness 的 supported_claims / partial_supported_claims / unsupported_claims。
- faithfulness 只负责“证据约束是否过线”，Judge 只负责“机制讲得是否好”。

## 3. 推荐权重

建议总分权重：

- 事实约束层：20%
- 机制深度层：50%
- 机制方向与一致性：20%
- 临床可用层：10%

说明：

- 若事实约束层低于阈值（例如 strict_sentence_grounding < 0.2），可触发总分上限惩罚。
- 其余部分按机制能力主导排序。

## 4. Judge 评分 rubric（0-2）

建议在当前 judge 维度基础上新增如下字段（0=差, 1=部分满足, 2=满足）：

- mechanism_chain_completeness
- mechanism_direction_correctness
- mechanism_granularity
- mechanism_internal_consistency
- uncertainty_calibration
- clinical_actionability

推荐不再保留旧版 label_alignment / sentence_grounding / non_hallucination 等维度，因为它们与 faithfulness 的职责重叠。

## 5. 汇总输出字段建议

建议新增 summary 字段（与现有 judge_summary 并行）：

- mechanism_chain_completeness_mean
- mechanism_direction_correctness_mean
- mechanism_granularity_mean
- mechanism_internal_consistency_mean
- uncertainty_calibration_mean
- clinical_actionability_mean
- mechanism_weighted_score_mean

建议不再保留旧版兼容字段作为主结果；如果为了历史报表兼容，可以单独映射到新字段，不应继续作为解释主口径。

## 6. 与当前代码的对接建议

当前已有能力：

- faithfulness 输出结构化 claim 诊断（supported/partial/unsupported）。
- judge 已改为机制导向 rubric，并与 faithfulness 解耦。

推荐改造顺序：

1. 先更新 judge prompt 与 result_normalizer，移除旧的 faithfulness/label 依赖。
2. 在 workflows.run_judge 的 summary 中只聚合机制维度均值。
3. 更新 README 与报告模板，统一结果解读口径。
4. 用小规模人工金标准集（200-300条）做机制链条相关性校准。

## 7. 报告口径建议

报告时分两张表：

- 可靠性表：strict grounding、unsupported率、query覆盖。
- 机制能力表：链条完整性、方向正确性、粒度、一致性。

避免只用一个 hallucination_rate 对模型优劣下结论。
