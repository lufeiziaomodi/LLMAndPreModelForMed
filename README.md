# LLM & PreModel for Med — DDI 机制可解释推理

> 面向药物-药物相互作用（DDI）识别的**机制可解释推理**研究项目。
> 核心链路：**PrimeKG 机制证据 → 外部大模型蒸馏 output → Llama3 LoRA 微调 → 推理 → 双通道评估（faithfulness × LLM-as-Judge）**。

---

## 1. 项目一览

**你可以用这个仓库做什么：**

| 场景 | 入口 | 说明 |
|---|---|---|
| 从原始 CSV 一路构造微调数据集 | `data_process/*.py` | rbert.csv → labels → KG → finetune JSON |
| **外部大模型直接对 test 集推理并评估**（新） | `apps/run_external_direct_eval.py` | 跳过训练，给微调后模型作 baseline |
| LoRA 微调 + LoRA 推理 + 评估 | `apps/run_experiment.py` | 训练 → 推理 → faithfulness + judge，一条命令 |
| **推理带 Agent Loop 反思-重试**（新） | `--enable_agent_loop` 或 YAML `agent_loop.enabled: true` | 事实度/机制分低于阈值时把 gap 反馈进上下文再生成 |
| 只跑评估（推理产物已有） | `apps/run_eval.py` | 复用 explanation_eval + judge |
| 多实验矩阵对比 | `apps/run_matrix.py` | 一个 config 跑多组变体 |
| 结果聚合 | `apps/aggregate_results.py` | 把各实验 compare_summary.csv 汇成一张表 |

**数据分层布局**：详见 [`docs/data_flow.md`](docs/data_flow.md)。

```
data/
├── raw/                DDIcorpus 源 CSV（进 git）
├── labels/             五类标签 JSON（extract_*_labels.py 产出）
├── kg/                 PrimeKG 图缓存（大文件 gitignore；drug_name_map.json 追踪）
├── finetune/{train,test,aux}/   微调数据集（进 git）
└── reports/{exp}/      推理与评估产物（进 git，跨机器同步）

results/                只放 LoRA adapter 权重（整体 gitignore）
```

---

## 2. 实验机快速开始（clone → 3 条命令跑通）

### 2.1 前置条件

- Python 3.10+；GPU 显存 ≥ 24GB（Llama3-8B QLoRA）
- 你需要在实验机本地准备好（不进 git 的部分）：
  - 基座模型：`models/Meta-Llama-3-8B-Instruct/`
  - PrimeKG 源文件：`data/kg/kg.csv`（PrimeKG 官方仓库下载）
  - DashScope API Key（用于蒸馏 / judge）：`export DASHSCOPE_API_KEY=sk-xxx`

### 2.2 首次同步（一次性）

```bash
git clone https://github.com/lufeiziaomodi/LLMAndPreModelForMed.git
cd LLMAndPreModelForMed

# 装依赖
pip install -r requirements.txt   # 或 conda activate <你的环境>

# 把大文件就位（本机手工放）
mkdir -p models data/kg
cp /path/to/kg.csv data/kg/kg.csv
cp -r /path/to/Meta-Llama-3-8B-Instruct models/

# 一次性构建 KG 查询图（≈ 数分钟）
python -m data_process.build_prime_kg
# 产出 data/kg/primekg_graph.pkl
```

### 2.3 三种典型实验，任选一个跑

#### A. 外部大模型直接评估（不训练，最快出结果）

```bash
export DASHSCOPE_API_KEY=sk-xxx
python -m apps.run_external_direct_eval \
    --config configs/experiments/external_direct_eval_baseline.yaml
```

当前 baseline 默认关闭 Agent Loop；反思-重试实验另行打开 `agent_loop.enabled`。若环境未安装 `PyYAML`，可改用等价的 `external_direct_eval_baseline.json` 配置。

产物：`data/reports/external_direct_eval_qwen_plus_no_kg/`
- `inference/test_out.json` — 外部 LLM 直接产出的解释
- `inference/agent_trace_*.json` — agent loop 每轮反思轨迹（开启循环时）
- `explanation_eval/faithfulness_summary_*.json` — 事实度指标
- `judge_eval/judge_summary_*.json` — LLM-as-judge 机制分

#### B. LoRA 微调 + 推理 + 评估（完整链路）

```bash
# 1. 蒸馏：qwen-plus 为训练集生成 output（首次或数据更新时才跑）
python data_process/generate_outputs_distillation.py \
    --config configs/experiments/restart_explanation_bootstrap.yaml \
    --config_section distillation_generation \
    --overwrite

# 2. 训练 + 推理 + 评估（一条命令）
python -m apps.run_experiment \
    --config configs/experiments/restart_explanation_bootstrap.yaml
```

产物：`data/reports/restart_reasoning_without_kg_bootstrap/` + `results/llama3_ddi_lora_reasoning_without_kg/`

#### C. 只补跑推理 + Agent Loop（已有 LoRA 权重）

```bash
python inference_llama3_ddi.py \
    --base_model models/Meta-Llama-3-8B-Instruct \
    --adapter_dir results/llama3_ddi_lora_reasoning_without_kg \
    --input_json data/finetune/test/input_test_reasoning_without_kg.json \
    --labels_json data/finetune/test/input_test_labels_clean.json \
    --output_json data/reports/my_exp/inference/test_out.json \
    --metrics_json data/reports/my_exp/inference/test_metrics.json \
    --prompt_mode explanation_without_kg \
    --enable_agent_loop \
    --agent_max_rounds 3 \
    --faith_min_coverage 0.4 \
    --faith_max_hallucination 0.5
```

---

## 3. Agent Loop（反思-重试推理机制）

**动机**：无论外部 LLM 还是 LoRA 微调后的 Llama3，第一次生成时都可能：
- 编造句子里没有的机制细节（faithfulness 低）
- 机制链断裂 / 方向反转 / 粒度不够（judge 低）

**做法**：对每一条 test 样本
1. 生成一次
2. 用 `FaithfulnessCritic`（本地免费）打分；可选叠加 `JudgeCritic`（调 qwen-max）
3. 如果 `coverage_ratio < 阈值` 或 `hallucination_rate > 阈值`，把不通过的 claim 拼进 messages 作 user turn 反馈：
   ```
   Your previous answer did not meet quality thresholds. Fix the issues below…
   - UNSUPPORTED by evidence: <claim1>
   - PARTIALLY supported (needs tightening): <claim2>
   Rewrite so that every mechanism claim is directly supported…
   ```
4. 重新生成，最多 N 轮
5. 全量 trace（每轮 output / critique / feedback）落 `data/reports/{exp}/inference/agent_trace_*.json`

**开关**：
- 外部直接评估：YAML `agent_loop.enabled: true`
- LoRA 推理：`--enable_agent_loop` CLI 参数
- 关键阈值：`faith_min_coverage`（默认 0.4）、`faith_max_hallucination`（默认 0.5）、`judge_min_overall_score`（默认 7.0/12）

详见 [`docs/agent_loop.md`](docs/agent_loop.md)。

---

## 4. 配置文件手册（YAML 字段速查）

配置根节点固定为四段 `experiment` / `training` / `inference` / `explanation_eval`（+ 可选 `agent_loop` / `judge` / `distillation_generation` / `external_generation` / `matrix`）。

### 4.1 通用

```yaml
experiment:
  name: my_experiment          # 报告落 data/reports/{name}/
  output_root: data/reports    # 保持默认即可
```

### 4.2 训练 / 推理

```yaml
training:
  enabled: true
  script: finetune_llama3_ddi.py
  args:                        # 直接映射为 --key value CLI
    - --data_path
    - data/finetune/train/input_reasoning_without_kg.json
    - --output_dir
    - results/llama3_ddi_lora_reasoning_without_kg   # LoRA 落 results/（gitignore）
    - --prompt_mode
    - explanation_without_kg
    - --max_steps
    - "1000"

inference:
  enabled: true
  script: inference_llama3_ddi.py
  args:
    - --input_json
    - data/finetune/test/input_test_reasoning_without_kg.json
    - --labels_json
    - data/finetune/test/input_test_labels_clean.json
    - --output_json
    - data/reports/my_experiment/inference/test_out.json
    - --metrics_json
    - data/reports/my_experiment/inference/test_metrics.json
    - --prompt_mode
    - explanation_without_kg
    # 可选：--enable_agent_loop
```

### 4.3 外部直接评估

```yaml
external_generation:
  input: data/finetune/test/input_test_reasoning_without_kg.json
  model: qwen-plus
  api_key: ${DASHSCOPE_API_KEY}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  generation_mode: explanation_without_kg   # 见 data_process/generate_outputs_distillation.py
  ignore_kg_evidence: true
  max_new_tokens: 768
  limit: null                  # smoke 时可设 20
```

### 4.4 Agent Loop

```yaml
agent_loop:
  enabled: true
  max_rounds: 3
  faithfulness_min_coverage: 0.4
  faithfulness_max_hallucination: 0.5
  use_judge_critic: false           # 开就多花 qwen-max 调用费
  judge_min_overall_score: 7.0
  keep_full_messages_in_trace: false
  verbose: true
```

### 4.5 事实度评估 + LLM-as-Judge

```yaml
explanation_eval:
  enabled: true
  model_name: my_experiment
  predictions_file: data/reports/my_experiment/inference/test_out.json
  labels_file: data/finetune/test/input_test_labels_clean.json
  output_dir: data/reports/my_experiment/explanation_eval

judge:
  enabled: true
  model_id: qwen-max
  api_key: ${DASHSCOPE_API_KEY}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  output_dir: data/reports/my_experiment/judge_eval
  max_retries: 4
  retry_delay: 2.0
```

**指标口径详解**：见 [`docs/evaluation_metrics_zh.md`](docs/evaluation_metrics_zh.md)。

---

## 5. 现有实验配置一览

| 配置文件 | 场景 |
|---|---|
| `configs/experiments/external_direct_eval_baseline.yaml` | 外部 qwen-plus 直接对 test 推理 + 评估（Agent Loop 可选） |
| `configs/experiments/restart_explanation_bootstrap.yaml` | 蒸馏 + LoRA 微调（reasoning_without_kg 版本） |
| `configs/experiments/restart_without_kg_no_lora_bootstrap.yaml` | 无 LoRA baseline：直接跑 Llama3 base + explanation_eval |
| `configs/experiments/restart_classification_only.yaml` | 只做五分类，不生成解释 |
| `configs/experiments/template_dual_track.yaml` | 双通道模板 |
| `configs/experiments/matrix_dual_track.yaml` | 矩阵对比 |

---

## 6. 进阶文档索引

- [`docs/data_flow.md`](docs/data_flow.md) — 数据流转 SSOT（分层结构、脚本用法、迁移路径）
- [`docs/agent_loop.md`](docs/agent_loop.md) — Agent Loop 设计与调优
- [`docs/evaluation_metrics_zh.md`](docs/evaluation_metrics_zh.md) — 事实度与 Judge 指标口径
- [`docs/mechanism_evaluation_framework_zh.md`](docs/mechanism_evaluation_framework_zh.md) — 机制导向评估框架 4 层权重
- [`docs/project_progress_current.md`](docs/project_progress_current.md) — 项目当前进展与卡点
- [`docs/simple_research_report_20260418.md`](docs/simple_research_report_20260418.md) — 阶段性研究报告

---

## 7. 常见问题

| 问题 | 解决 |
|---|---|
| `FileNotFoundError: data/kg/primekg_graph.pkl` | 执行 `python -m data_process.build_prime_kg`（需要先手工放 `data/kg/kg.csv`） |
| `--use_judge_critic requires DASHSCOPE_API_KEY` | `export DASHSCOPE_API_KEY=sk-xxx` |
| 训练卡住/OOM | 减小 `--batch_size`、加大 `--grad_accum`；推理端已内置 OOM 自动降级（缩短 kg_evidence / max_new_tokens） |
| 只想 smoke 5 条样本 | 外部直评：`external_generation.limit: 5`；LoRA 推理：`--limit 5` |
| Agent Loop 一直不通过 | 把 `faith_min_coverage` 降到 0.3、`faith_max_hallucination` 提到 0.6；用 `--verbose` 观察每轮 critique 原因 |
| 推理产物 predicted_label 为空 | `explanation_only` 模式本身不吐 label；在 `explanation_eval` 里配 `label_predictions_file` 从 label-only 那次结果回填 |
| 想切换到另一个模型 baseline | 复制 `external_direct_eval_baseline.yaml`，改 `external_generation.model`、`experiment.name` 即可 |

---

## 8. 目录导航

```
apps/                      实验入口（外部直评、训练、推理、评估、矩阵、聚合）
configs/experiments/       YAML 配置
data/                      数据分层（详见 docs/data_flow.md）
data_process/              数据构造脚本 + paths.py 统一路径
docs/                      文档
evaluate/                  事实度 + Judge 评估
finetune_llama3_ddi.py     LoRA 微调
inference_llama3_ddi.py    LoRA 推理（支持 --enable_agent_loop）
pipelines/                 编排层
  ├── agent_loop.py            Agent Loop 反思-重试引擎（新增）
  ├── experiment_pipeline.py   完整实验流程
  ├── explanation_pipeline.py  评估阶段
  └── ...
prime_kg_utils/            PrimeKG 查询工具
results/                   LoRA adapter 权重（整体 gitignore）
```

---

## License

MIT.

## Disclaimer

本项目仅供研究和教育目的使用，不应用于临床实践或医疗决策。所有预测结果应结合专业医学知识进行评估。
