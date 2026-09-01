# 下一阶段执行计划：With-KG 基线、Agent Loop 与论文起稿

更新时间：2026-09-01

## 1. 当前可确认的研究状态

### 已有真实结果

- **Llama3 Base No-KG**：2026-04 历史全量实验，1042 条。该结果使用较早评估口径，适合保留作历史参照，不宜与新版 Judge 直接并列下结论。
- **Llama3 LoRA Reasoning No-KG**：2026-04 已完成训练、推理和评估；同样属于旧评估口径，主要证明“蒸馏微调链路可运行”。
- **Qwen-Plus Direct No-KG**：完成 1042 条全量直接推理，0 条生成失败。按 2026-09-01 校准口径重算后，Faithfulness：strict support 0.0291、partial support 0.5162、hallucination 0.4548、query coverage 0.8732；Judge：机制总分 8.4904/12，good rate 0.2831。
- **Qwen-Plus Direct Raw-KG**：2026-09-01 完成 1042 条全量直接推理，0 条生成失败。Faithfulness：strict support 0.0128、partial support 0.6617、hallucination 0.3255、query coverage 0.8764、KG grounded 0.3714；Judge 有效 1039 条，机制总分 8.5938/12、good rate 0.2964。3 条最长 Raw-KG 样本因上下文达到 7.4 万至 10.5 万字符而被 qwen-max 以 HTTP 400 拒绝，失败记录保留但不计入有效均值。

### 已落地但尚未全量验证的能力

- `query_group` 兼容、KG 证据输入、结构化 explanation 解析。
- Faithfulness 与 LLM-as-Judge 解耦的双通道评价。
- 外部模型直评入口，可直接生成、评价并落盘比较表。
- Agent Loop 的“生成—批评—反馈—重试”实现和 trace。

### 不能当作实测结论的内容

- LoRA With-KG 的效果水位仍是 PPT 中的 Mock 预期，不是实验结果。
- Agent Loop 尚无统一测试集的真实收益、成本或收敛轮次数据。
- 新版句子条件 KG Evidence Summary 尚未重建为正式训练/测试数据；当前 With-KG 测试集仍是基于 `queries` 的旧式一阶邻居证据。

## 2. 下一步实验顺序

### 实验 A：Qwen-Plus Direct With-KG 全量基线（已完成）

目的：在与已完成 No-KG 基线相同的模型、样本数和评价体系下，测量 KG 是否提高证据支撑和机制粒度。

- 配置：`configs/experiments/external_direct_eval_with_kg.json`
- 输入：`data/finetune/test/input_test_label_only_with_kg.json`，1042 条。
- 输出协议：一句话现象解释；先识别代谢/吸收等 cue，再验证 CYP 等 KG 锚点；最多保留 3 个代表药物对。
- 报告重点：coverage、hallucination、query coverage、KG grounded、机制总分、链条完整性、机制粒度。
- 阶段结论：Raw-KG 将 hallucination 降低 0.1292、partial support 提高 0.1455、Judge 提高 0.1034（按有效样本），但 strict support 降低 0.0163。原始邻居有助于提供机制锚点，却不能保证完整因果链受到证据支持；极端长上下文还会直接触发 API 拒绝。这一结果为后续 Anchor Filter 和 Cue-conditioned Evidence Summary 提供了必要动机。

### 实验 B：Qwen-Plus Direct With-KG + Faithfulness Agent Loop

目的：只在事实约束未通过时重试，评估“质量增益—额外调用次数”的性价比。

- 配置：`configs/experiments/external_direct_eval_with_kg_agent_faith.json`
- 与实验 A 保持同一模型、同一输入、同一评价；差异仅为 `agent_loop.enabled=true`。
- 额外统计：首轮通过率、最终通过率、平均轮次、`max_rounds` 占比、每提升 0.01 coverage 的额外调用次数。
- 第一轮只使用本地 Faithfulness Critic；Judge Critic 放到后续小规模消融，避免一开始引入双重 API 成本。

### 实验 C：KG Evidence Summary 重建与 LoRA With-KG

目的：把“外部模型是否能用 KG”推进为“本地模型能否蒸馏到这种解释能力”。

1. 将 KG 由原始一阶邻居改为句子条件下的 Evidence Summary。
2. 生成 reasoning_with_kg 训练集和测试集。
3. 先对 50 条人工抽样检查输出与证据一致性，再做全量蒸馏、LoRA 和统一评估。
4. 与实验 A、B、No-KG LoRA 在同一评估口径下对比。

## 3. Windows 运行命令

在新的 PowerShell 窗口中必须先进入仓库根目录，否则会出现 `No module named apps`：

```powershell
Set-Location C:\Project\LLMAndPreModelForMed
$env:DASHSCOPE_API_KEY = "<你的 DashScope Key>"

# A. 先跑 With-KG 外部直接基线
py -3.10 -m apps.run_external_direct_eval `
  --config configs/experiments/external_direct_eval_with_kg.json

# B. A 完成并核对结果后，再跑 Agent Loop 对照
py -3.10 -m apps.run_external_direct_eval `
  --config configs/experiments/external_direct_eval_with_kg_agent_faith.json
```

若该终端没有 `py -3.10`，使用当前可用解释器：

```powershell
python -m apps.run_external_direct_eval `
  --config configs/experiments/external_direct_eval_with_kg.json
```

结果分别落在：

- `data/reports/external_direct_eval_qwen_plus_with_kg/`
- `data/reports/external_direct_eval_qwen_plus_with_kg_agent_faith/`

## 4. 论文现在可以起草的部分

实验 A 已可写入结果章节；实验 B 运行期间可并行开始写方法章节和 Raw-KG 误差分析。不要将 Mock 水位写进结果章节。

1. **引言**：DDI 分类的局限在于只给标签而不解释机制；医学场景还要求可追溯、可核验。
2. **方法**：双轨任务划分（分类与解释分离）、句子条件 KG 锚点、现象级结构化输出、Faithfulness × Judge 双通道评价、Agent Loop 的质量门控。
3. **实验设置**：统一 1042 条测试集、Qwen-Plus Direct No-KG / With-KG、Agent Loop 消融、评价指标和成本统计。
4. **结果与讨论**：先写 No-KG 与 Raw-KG 全量对照；Agent Loop 的质量—成本曲线待实验 B 后补充。
5. **误差分析**：挑选“无可靠 KG 锚点”“方向不清”“代表药物对遗漏”“重复重试无收益”四类案例。

## 5. 当前验收标准

- 实验 A：已验收。1042 条生成无空输出，Faithfulness 与 Judge 报告完整落盘；3 条超长 Judge 失败已单独记录。
- 实验 B：trace 完整；可统计平均轮次与最终通过率；报告质量增益是否值得额外成本。
- 论文：方法与实验设置章节先完成，结果章节仅引用可追溯报告文件中的实测数值。
