# 评估指标详解（Faithfulness × LLM-as-Judge）

> 从旧版 README 抽取。对应代码：`evaluate/explanation/faithfulness.py`、
> `evaluate/explanation/processor.py`、`evaluate/judge/qwen_judge.py`、
> `evaluate/judge/judge_prompts.py`。

## 1. explanation_eval（faithfulness）

当前版本的 faithfulness 不再把整段 JSON 输出当自然语言切分，而是先做结构化解析，再只评估真正的解释性字段。

### 1.1 claim 抽取规则

- 解释输出优先按 JSON 解析。
- 仅抽取 `analysis_steps` 与 `mechanism_summary` 中的文本作为 claim 候选。
- `query`、`representative_pairs`、JSON 括号、字段名、编号（如 `1. 2. 3.`）等结构性内容不参与 hallucination 统计。
- 如果输出不是合法 JSON，才退化到纯文本切分。

### 1.2 证据来源与 grounding 口径

- `no_kg`：证据仅包含输入句子本身。
- `with_kg`：证据包含输入句子 + `kg_evidence`。
- PrimeKG 在当前项目中的作用是"机制锚点"而不是"完整关系语义"。图谱边通常只表示"药物与酶/转运体/靶点存在关联"，不直接给出"抑制/诱导/底物代谢"等细粒度关系。
- 因此，当前 faithfulness 允许两类相对合理的支持来源：
  - 句子中的显式事实或近似改写。
  - 被句子/KG 锚点约束住的机制性推理。
- 但如果解释新增了更具体的机制细节，而这些细节既不在句子中，也没有被当前 KG 证据显式支撑，则仍会被判为 `partial` 或 `unsupported`，而不是 `supported`。

### 1.3 评分层级

- `supported_claims`：被当前证据直接支持，或属于较强的句内改写/证据锚定改写。
- `partial_supported_claims`：有明显证据锚点，但包含一定概括、抽象或外推成分。
- `unsupported_claims`：缺乏当前证据支持，或引入了额外机制细节、外部医学知识、过强结论。

### 1.4 指标含义

- `coverage_ratio`：`supported_claims / all_claims`。表示"完全支持"的解释比例。
- `partial_support_rate`：`partial_supported_claims / all_claims`。表示"部分支持/锚定推理"的比例。
- `hallucination_rate`：`unsupported_claims / all_claims`。表示当前证据下不被支持的解释比例。
- `consistency_score`：预测标签与解释文本的一致性分数（关键词命中 + 反例惩罚）。范围 [0,1]。
- `kg_grounded_ratio`：仅在 `with_kg` 下计算，claim 被 KG 证据单独支持的比例。
- `coverage_mean` / `partial_support_mean` / `hallucination_mean` / `consistency_mean` / `kg_grounded_mean`：上述指标的样本均值。
- `by_evidence_mode`：按 `with_kg` 与 `no_kg` 分桶后的均值指标。

### 1.5 当前 faithfulness 设计意图

- 当前版本故意比旧版更严格。
- 目标不是奖励"医学上合理"，而是区分：
  - 当前输入已经表达或锚定的内容。
  - 需要外部知识和模型推理补出的内容。
- 对于 `no_kg` 解释，`CYP2C9 inhibition`、`potential hypoglycemia`、`enterohepatic recirculation` 这类机制补充，如果句子里没有明确说出，通常不会进入 `supported_claims`。
- 对于 `with_kg` 解释，如果 claim 与图谱中的酶/转运体/靶点锚点存在较强重叠，则会比 `no_kg` 更容易进入 `supported` 或 `partial`。

## 2. judge_eval（LLM-as-judge）

### 2.1 输入

Judge 现在只评估机制质量，不再读取 faithfulness 诊断结果。其输入包含：

- `sentence`
- `queries`
- 原始 `explanation`
- 可选 `kg_evidence`

Judge 不再依赖 `supported_claims`、`partial_supported_claims`、`unsupported_claims`、`query_coverage`。

### 2.2 评分维度（0-2 rubric × 6 维度）

- `mechanism_chain_completeness`：机制链是否完整
- `mechanism_direction_correctness`：方向与因果是否正确
- `mechanism_granularity`：机制粒度是否足够具体
- `mechanism_internal_consistency`：内部是否自洽
- `uncertainty_calibration`：是否会在证据不足时表达不确定性
- `clinical_actionability`：是否给出可执行临床含义

汇总：

- `mechanism_gaps`：机制缺口列表
- `judge_short_rationale`：简短理由
- `mechanism_overall_score`：六项 0-2 维度的总分，范围 [0,12]
- `mechanism_overall_decision`：`poor` / `fair` / `good`

### 2.3 严格口径

- Judge 的目标不是判断是否"被句子直接支持"，而是判断解释是否真的把机制讲清楚。
- 当解释只重复标签、只罗列实体名、或只有泛化结论而没有机制链时，应显著降分。
- 当解释给出过细的机制结论但前后不一致、方向自相矛盾或缺乏不确定性表达时，也应降分。

## 3. 两类指标关系

- `explanation_eval` 主要是 [0,1] 比例分数。
- `judge_eval` 主要是 [0,2] 机制 rubric 与 [0,12] 总分。
- 两类指标不建议直接横向比绝对值，应在各自量纲内比较趋势与相对变化。

## 4. 推荐 4 层权重（机制导向框架）

- 事实约束层：20%
- 机制深度层：50%
- 机制方向与一致性：20%
- 临床可用层：10%

详细 rubric、字段定义与落地步骤见 [`mechanism_evaluation_framework_zh.md`](mechanism_evaluation_framework_zh.md)。

## 5. 评估产物排查建议

优先检查 `faithfulness_detail_*.json` 中以下字段：

- `predicted_label`
- `predicted_label_source`
- `supported_claims`
- `partial_supported_claims`
- `unsupported_claims`

如果出现以下现象，通常表示评估口径仍需调整：

- `predicted_label` 大量为空：说明 label 回填链路未接上。
- `unsupported_claims` 出现 `[`、`{`、`"query"`、`representative_pairs`：说明结构化 claim 抽取失效。
- `coverage_mean` 极低且 `partial_support_mean` 很高：说明口径可能过严，更多奖励了"锚定推理"而不是"完全支持"。
- `hallucination_mean` 异常偏低，且 judge 总体很乐观：说明 Judge 仍可能过宽松。
