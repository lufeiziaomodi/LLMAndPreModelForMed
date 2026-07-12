# MedGemma-27B-IT-8bit DDI关系识别微调指南

本指南提供了使用LoRA技术对MedGemma-27B-IT-8bit模型进行微调的完整流程，以实现药物-药物相互作用(DDI)关系识别任务。

## 项目最新进展（2026-04）

当前项目已完成从“蒸馏 -> 训练 -> 推理 -> 解释评估”的统一流程改造，核心进展如下：

- 蒸馏脚本已统一命名为 `data_process/generate_outputs_distillation.py`（不再使用 provider 绑定命名）。
- 蒸馏配置段名统一为 `distillation_generation` 与 `distillation_generation_smoke`。
- `restart_reasoning_without_kg_bootstrap` 实验采用 `qwen-plus` 进行无 KG 推理蒸馏。
- 推理结果、解释评估、Judge 评估路径已收敛到实验名目录下（`output_root/experiment_name`）。
- `restart_reasoning_without_kg_bootstrap` 当前状态：训练完成、推理完成、explanation_eval 完成；judge 可独立补跑。

## 当前推荐流程（统一实验链路）

### 0) 环境

建议使用项目实验环境：

```bash
conda activate ddi_llm_ft
```

### 1) 蒸馏（qwen-plus）

先 smoke 验证：

```bash
python data_process/generate_outputs_distillation.py \
    --config configs/experiments/restart_explanation_bootstrap.yaml \
    --config_section distillation_generation_smoke \
    --overwrite
```

再全量蒸馏：

```bash
python data_process/generate_outputs_distillation.py \
    --config configs/experiments/restart_explanation_bootstrap.yaml \
    --config_section distillation_generation \
    --overwrite
```

### 2) 训练 + 推理 + 评估（统一入口）

```bash
python -m apps.run_experiment --config configs/experiments/restart_explanation_bootstrap.yaml
```

说明：在当前工程结构下，建议优先使用 `python -m apps.run_experiment`。

## 最新输出目录约定

`output_root` 保持为 `results/eval_report`，并以实验名作为一级子目录，例如：

```text
results/eval_report/restart_reasoning_without_kg_bootstrap/
├── finetune_dataset_input_test_out_reasoning_without_kg.json
├── finetune_dataset_input_test_metrics_reasoning_without_kg.json
├── explanation_eval/
│   ├── faithfulness_detail_*.json
│   └── faithfulness_summary_*.json
└── judge_eval/
        ├── judge_detail_*.json
        └── judge_summary_*.json
```

## explanation_eval 与 judge_eval 指标中文释义

以下释义对应当前实现（`evaluate/explanation/faithfulness.py`、`evaluate/explanation/processor.py`、`evaluate/judge/qwen_judge.py`、`evaluate/judge/judge_prompts.py`）。

### explanation_eval（faithfulness）

当前版本的 faithfulness 不再把整段 JSON 输出直接当自然语言切分，而是先做结构化解析，再只评估真正的解释性字段。

#### 1) claim 抽取规则

- 解释输出优先按 JSON 解析。
- 仅抽取 `analysis_steps` 与 `mechanism_summary` 中的文本作为 claim 候选。
- `query`、`representative_pairs`、JSON 括号、字段名、编号（如 `1. 2. 3.`）等结构性内容不参与 hallucination 统计。
- 如果输出不是合法 JSON，才退化到纯文本切分。

#### 2) 证据来源与 grounding 口径

- `no_kg`：证据仅包含输入句子本身。
- `with_kg`：证据包含输入句子 + `kg_evidence`。
- PrimeKG 在当前项目中的作用是“机制锚点”而不是“完整关系语义”。也就是说，图谱边通常只表示“药物与酶/转运体/靶点存在关联”，不直接给出“抑制/诱导/底物代谢”等细粒度关系。
- 因此，当前 faithfulness 允许两类相对合理的支持来源：
  - 句子中的显式事实或近似改写。
  - 被句子/KG 锚点约束住的机制性推理。
- 但如果解释新增了更具体的机制细节，而这些细节既不在句子中，也没有被当前 KG 证据显式支撑，则仍会被判为 `partial` 或 `unsupported`，而不是 `supported`。

#### 3) 评分层级

- `supported_claims`：被当前证据直接支持，或属于较强的句内改写/证据锚定改写。
- `partial_supported_claims`：有明显证据锚点，但包含一定概括、抽象或外推成分。
- `unsupported_claims`：缺乏当前证据支持，或引入了额外机制细节、外部医学知识、过强结论。

#### 4) 指标含义

- `coverage_ratio`：`supported_claims / all_claims`。表示“完全支持”的解释比例。
- `partial_support_rate`：`partial_supported_claims / all_claims`。表示“部分支持/锚定推理”的比例。
- `hallucination_rate`：`unsupported_claims / all_claims`。表示当前证据下不被支持的解释比例。
- `consistency_score`：预测标签与解释文本的一致性分数（关键词命中 + 反例惩罚）。范围 [0,1]。
- `kg_grounded_ratio`：仅在 `with_kg` 下计算，claim 被 KG 证据单独支持的比例。
- `coverage_mean` / `partial_support_mean` / `hallucination_mean` / `consistency_mean` / `kg_grounded_mean`：上述指标的样本均值。
- `by_evidence_mode`：按 `with_kg` 与 `no_kg` 分桶后的均值指标。

#### 5) 当前 faithfulness 设计意图

- 当前版本故意比旧版更严格。
- 它的目标不是奖励“医学上合理”，而是区分：
  - 当前输入已经表达或锚定的内容。
  - 需要外部知识和模型推理补出的内容。
- 对于 `no_kg` 解释，像 `CYP2C9 inhibition`、`potential hypoglycemia`、`enterohepatic recirculation` 这类机制补充，如果句子里没有明确说出，通常不会进入 `supported_claims`。
- 对于 `with_kg` 解释，如果 claim 与图谱中的酶/转运体/靶点锚点存在较强重叠，则会比 `no_kg` 更容易进入 `supported` 或 `partial`。

### judge_eval（LLM-as-judge）

#### 1) 当前 Judge 的输入

Judge 现在只评估机制质量，不再读取 faithfulness 诊断结果。其输入包含：

- `sentence`
- `queries`
- 原始 `explanation`
- 可选 `kg_evidence`

Judge 不再依赖 `supported_claims`、`partial_supported_claims`、`unsupported_claims`、`query_coverage` 或任何 faithfulness 输出。

#### 2) 评分维度

- `mechanism_chain_completeness`：机制链是否完整。整数 0-2。
- `mechanism_direction_correctness`：方向与因果是否正确。整数 0-2。
- `mechanism_granularity`：机制粒度是否足够具体。整数 0-2。
- `mechanism_internal_consistency`：内部是否自洽。整数 0-2。
- `uncertainty_calibration`：是否会在证据不足时表达不确定性。整数 0-2。
- `clinical_actionability`：是否给出可执行临床含义。整数 0-2。
- `mechanism_gaps`：Judge 识别出的机制缺口列表。
- `judge_short_rationale`：简短理由。
- `mechanism_overall_score`：六项 0-2 维度的总分，范围 [0,12]。
- `mechanism_overall_decision`：`poor` / `fair` / `good`。

#### 3) 当前 Judge 的严格口径

- Judge 的目标不是判断是否“被句子直接支持”，而是判断解释是否真的把机制讲清楚。
- 当解释只重复标签、只罗列实体名、或只有泛化结论而没有机制链时，应显著降分。
- 当解释给出过细的机制结论但前后不一致、方向自相矛盾或缺乏不确定性表达时，也应降分。

注意：

- explanation_eval 主要是 [0,1] 比例分数。
- judge_eval 主要是 [0,2] 机制 rubric 与 [0,12] 总分。
- 两类指标不建议直接横向比绝对值，应在各自量纲内比较趋势与相对变化。

## 评估模块更新（2026-04）

本轮更新的目标，是修复 explanation_eval / judge_eval 在结构化解释场景下的几个核心问题。

### 已完成更新

1. `predicted_label` 回填

- explanation-only 输出里常见 `predicted_label=""`。
- 当前评估支持通过 `label_predictions_file` 引入第一阶段 label-only 结果进行回填。
- 若未显式配置，评估器会尝试根据 `predictions_file` 文件名自动推断对应的 `label_only_*` 结果路径。
- 回填优先使用 `(sentence, queries)` 键匹配，避免索引漂移；匹配失败时再退化为按索引对齐。

2. JSON 噪声移除

- 旧版会把 `[`、`{`、`"query"`、编号、字段名等都切成 claim，导致 hallucination 严重虚高。
- 当前版本先结构化解析 JSON，只保留真正的解释字段。

3. `representative_pairs` 不再计入 hallucination

- 该字段只表示查询覆盖或代表性 pair，不属于机制性 claim。
- 当前版本不会把它纳入 faithfulness 统计。

4. 句外机制推断不再被误判为 fully supported

- 对 `no_kg` 输出，如果解释补充了句外的酶机制、代谢方式、不良反应风险等，通常会落到 `partial` 或 `unsupported`。
- 这避免了旧版“只要医学上合理就算支持”的偏宽口径。

5. Judge 提示词与输入更新

- Judge prompt 已改成 strict evidence-grounded 口径。
- Judge 同时参考 deterministic claim diagnostics，而不只看原始解释文本。

### 当前建议的评估运行方式

#### 1) 推荐在配置中显式给出 label-only 结果

```yaml
explanation_eval:
  enabled: true
  predictions_file: results/eval_report/your_exp/finetune_dataset_input_test_out_reasoning_without_kg.json
  labels_file: data/finetune_dataset_input_test_labels_clean.json
  label_predictions_file: results/test_predictions/finetune_dataset_input_test_out_label_only_without_kg.json
  output_dir: results/eval_report/your_exp/explanation_eval
```

#### 2) 只跑评估

```bash
python apps/run_eval.py --config configs/experiments/your_experiment.yaml
```

#### 3) 推荐的补跑顺序

1. 先完成推理，得到 explanation/reasoning 输出。
2. 准备第一阶段 label-only 结果，用于 `predicted_label` 回填。
3. 跑 explanation_eval，检查 `faithfulness_detail_*.json`。
4. 确认 claim 解析、回填标签、hallucination 口径正常后，再补跑 judge。

### 评估产物排查建议

优先检查 `faithfulness_detail_*.json` 中以下字段：

- `predicted_label`
- `predicted_label_source`
- `supported_claims`
- `partial_supported_claims`
- `unsupported_claims`

如果出现以下现象，通常表示评估口径仍需调整：

- `predicted_label` 大量为空：说明 label 回填链路未接上。
- `unsupported_claims` 出现 `[`、`{`、`"query"`、`representative_pairs`：说明结构化 claim 抽取失效。
- `coverage_mean` 极低且 `partial_support_mean` 很高：说明口径可能过严，更多奖励了“锚定推理”而不是“完全支持”。
- `hallucination_mean` 异常偏低，且 judge 总体很乐观：说明 Judge 仍可能过宽松。

## 机制导向评估框架（新增建议）

当前项目目标是“机制挖掘优先”，而不仅是标签一致。因此建议把评估目标拆成三层：

- 事实约束层（底线）：检查句证支持与明显幻觉，防止胡编。
- 机制深度层（主指标）：检查机制链是否完整、方向是否正确、粒度是否足够。
- 临床可用层（加分项）：检查是否给出风险类型与可执行建议。

推荐总分权重：

- 事实约束层：20%
- 机制深度层：50%
- 机制方向与一致性：20%
- 临床可用层：10%

该框架与现有 `faithfulness` / `judge` 不冲突：

- `faithfulness` 继续做严格证据约束。
- `judge` 增加机制链条维度，作为主评估通道。

详细 rubric、字段定义与落地步骤见：

- [docs/mechanism_evaluation_framework_zh.md](docs/mechanism_evaluation_framework_zh.md)

## 新增：统一实验架构入口（推荐）

为了支持分类链路与解释链路解耦、KG消融、蒸馏对比和LLM-as-judge评测，项目新增了统一的实验编排层：

- `apps/`：统一入口脚本（训练、评估、全流程）
- `pipelines/`：流程编排层（classification/explanation/training/compare）
- `configs/`：可复用实验配置
- `evaluate/`：模块化评估系统（含 qwen-max judge）

### 推荐运行方式

1) 使用统一全流程入口：

```bash
python apps/run_experiment.py --config configs/experiments/template_dual_track.yaml
```

2) 仅运行评估：

```bash
python apps/run_eval.py --config configs/experiments/template_dual_track.yaml
```

2.1) 运行矩阵对比实验：

```bash
python apps/run_matrix.py --config configs/experiments/matrix_dual_track.yaml
```

3) 仅运行训练：

```bash
python apps/run_train.py --config configs/experiments/template_dual_track.yaml
```

4) 聚合多个 run 的对比结果：

```bash
python apps/aggregate_results.py --root results/experiments --output results/experiments/aggregate_compare_summary.csv
```

### qwen-max Judge

如需启用 LLM-as-judge，请配置：

```bash
set DASHSCOPE_API_KEY=your_api_key
```

或在配置文件的 `judge.api_key` 中填写。

## 项目结构（模块化版本）

```
ddi_medgemma_ft/
├── data/
│   ├── rbert_train.csv              # 原始训练数据
│   └── processed_lora_data_base.csv # 处理后的LoRA训练数据
├── data_process/
│   ├── get_lora_data.py             # 高级数据处理脚本
│   └── get_lora_data_base.py        # 基础数据处理脚本
├── modules/                         # 模块化组件
│   ├── __init__.py                  # 包初始化文件
│   ├── data_utils.py                # 数据加载和预处理模块
│   ├── model_utils.py               # 模型加载和配置模块
│   ├── training_utils.py            # 模型训练模块
│   └── utils.py                     # 工具函数模块
├── finetune_medgemma_lora.py        # LoRA微调主脚本（模块化版本）
├── inference_medgemma_lora.py       # 推理脚本
├── requirements.txt                 # 项目依赖
└── README.md                        # 项目说明文档
```

## 模块化说明

项目已拆分为以下核心模块：

1. **data_utils.py** - 负责数据加载、预处理和格式化
2. **model_utils.py** - 负责模型加载、量化配置和LoRA设置
3. **training_utils.py** - 负责模型训练逻辑
4. **utils.py** - 提供辅助功能，如生成推理脚本和依赖文件

这种模块化设计使代码更易于维护、扩展和重用。各个模块职责明确，相互独立，可以单独修改或优化。

## 步骤1: 环境准备

### 创建虚拟环境

```bash
# 使用Anaconda创建虚拟环境
conda create -n medgemma python=3.10
conda activate medgemma

# 或者使用venv
python -m venv medgemma
# Windows激活
medgemma\Scripts\activate
# Linux/Mac激活
# source medgemma/bin/activate
```

### 安装依赖

```bash
# 安装PyTorch (根据您的CUDA版本选择)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

## 步骤2: 数据准备

确保您已经准备好了处理后的数据集：`data/processed_lora_data_base.csv`。该文件应当包含两列：
- `input_text`: 包含任务说明和带实体标记的文本
- `output_text`: 对应的DDI关系标签

如果需要重新处理数据，可以运行：

```bash
python data_process/get_lora_data_base.py
```

## 步骤3: 模型下载与配置

使用Hugging Face Transformers库自动下载模型：

```python
# 在代码中会自动处理下载
# 模型ID: google/medgemma-27b-it-8bit
```

### 注意事项

- 下载模型需要稳定的网络连接和足够的磁盘空间（约需35-40GB）
- 模型将在首次运行时自动下载并缓存
- 建议配置Hugging Face token以获得更高的下载限额

```bash
# 设置Hugging Face token (可选)
export HF_TOKEN="your_token_here"
```

## 步骤4: 执行LoRA微调

### 基本运行方式

```bash
python finetune_medgemma_lora.py
```

### 自定义参数运行

```bash
python finetune_medgemma_lora.py \
    --model_id "google/medgemma-27b-it-8bit" \
    --train_data "data/processed_lora_data_base.csv" \
    --output_dir "./results/medgemma_ddi_lora" \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_steps 1000 \
    --r 8 \
    --lora_alpha 16
```

### 参数说明

- `--model_id`: 模型ID
- `--train_data`: 训练数据路径
- `--output_dir`: 输出目录
- `--batch_size`: 每个设备的批次大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--max_steps`: 最大训练步数
- `--r`: LoRA注意力维度
- `--lora_alpha`: LoRA alpha参数

## 步骤5: 评估模型

训练完成后，脚本会自动执行评估并输出结果指标。

## 步骤6: 使用微调后的模型进行推理

### 基本推理

运行推理脚本：

```bash
python inference_medgemma_lora.py
```

### 自定义推理示例

修改`inference_medgemma_lora.py`中的示例输入：

```python
# 自定义示例输入
sample_input = """Task: Determine the DDI relationship type between <e1> and <e2> in the text. Only output the label (false/mechanism/effect/advise/int).
Text: 您的文本内容，包含<e1>和<e2>标记的实体"""
```

## 训练脚本主要功能说明

### 1. 数据加载与预处理

脚本自动加载CSV格式的数据集，并按比例划分为训练集和验证集。

### 2. 模型加载与量化

使用8-bit量化加载MedGemma-27B-IT模型，节省显存使用。

### 3. LoRA配置

配置LoRA参数，仅对模型的关键部分（注意力层和MLP层）进行微调，显著减少训练参数量。

### 4. 训练过程

使用SFTTrainer进行监督微调，支持梯度累积、学习率调度等优化技术。

### 5. 模型保存

训练完成后自动保存LoRA适配器权重，便于后续加载和使用。

## 硬件要求

- GPU内存：推荐使用至少24GB显存的GPU（如RTX 3090/4090、A10等）
- 系统内存：至少32GB
- 存储：至少50GB可用空间

## 常见问题与解决方案

### CUDA内存不足

- 减小batch_size
- 增加gradient_accumulation_steps
- 使用更小的LoRA秩(r值)

### 模型下载失败

- 检查网络连接
- 配置Hugging Face token
- 尝试手动下载模型文件

### 训练速度慢

- 使用更大的batch_size（如果显存允许）
- 增加gradient_accumulation_steps
- 考虑使用多GPU训练（脚本支持device_map="auto"）

## 高级配置

### 多GPU训练

脚本支持使用`device_map="auto"`自动分配模型到多个GPU上。

### 自定义LoRA目标层

可以在`configure_lora`函数中调整`target_modules`参数，以适应不同模型架构的需求。

### 自定义训练超参数

根据实际情况调整学习率、批量大小、训练步数等超参数，以获得最佳性能。

## 扩展与优化建议

1. **数据增强**：考虑对训练数据进行增强，如同义词替换、文本重构等
2. **超参数搜索**：使用网格搜索或贝叶斯优化寻找最佳超参数组合
3. **集成学习**：训练多个模型并进行集成，提高预测稳定性
4. **持续学习**：在新数据上继续微调，适应新的DDI识别需求

## 许可证

[MIT License](LICENSE)

## 免责声明

本项目仅供研究和教育目的使用，不应用于临床实践或医疗决策。所有预测结果应结合专业医学知识进行评估。
