# 数据流转与目录布局（2026-07 重构后）

> 本文档描述项目**数据从 DDIcorpus 源头 → 微调数据 → 模型训练/推理 → 评估报告**的完整链路，
> 以及重构后 `data/` 与 `results/` 的分层规范。
>
> 关联脚本：`data_process/paths.py`（所有路径常量的唯一源头）、`data_process/*.py`（各阶段构造脚本）。

---

## 1. 数据分层图

```
                     ┌────────────────────────────────────────────────┐
                     │  data/raw/                     [追踪 · <20MB]  │
                     │  DDIcorpus 源 CSV                              │
                     │  · rbert_train.csv / rbert_test.csv            │
                     │  · standardized_generated_rbert_train_random.csv│
                     │  · test_augmented_data.csv                     │
                     └───────────────┬────────────────────────────────┘
                                     │
                          extract_train_labels.py
                          extract_test_labels.py
                                     ▼
                     ┌────────────────────────────────────────────────┐
                     │  data/labels/                  [追踪]          │
                     │  五类标签 JSON                                 │
                     │  · mechanism.json / mechanism_test.json        │
                     │  · effect.json / effect_test.json              │
                     │  · advise.json  / advise_test.json             │
                     │  · int.json     / int_test.json                │
                     │  · false.json   / false_test.json              │
                     └───────────────┬────────────────────────────────┘
                                     │
                                     │  (需要 KG 证据的模式：还要读 data/kg/)
                                     ▼
              ┌────────────────────────────────────────────────────┐
              │  data/kg/                        [大文件不追踪]      │
              │  PrimeKG 资源（仅实验机本地存在）                    │
              │  · kg.csv                     ← 数百 MB（gitignore）│
              │  · primekg_graph.pkl          ← 数百 MB（gitignore）│
              │  · primekg_pyg.pt (optional)  ← 数百 MB（gitignore）│
              │  · drug_name_map.json         ← 小文件（追踪）        │
              │  build_prime_kg.py 产出 pkl/pt                      │
              └────────────────────┬───────────────────────────────┘
                                   │
                       build_finetune_dataset.py
                                   ▼
        ┌───────────────────────────────────────────────────────────┐
        │  data/finetune/                       [追踪 · ~50MB]       │
        │  ├── train/          训练用微调数据集                       │
        │  │    ├── input_label_only_with_kg.json                    │
        │  │    ├── input_label_only_without_kg.json                 │
        │  │    ├── input_reasoning_without_kg.json                  │
        │  │    └── input_labels.json (可选 sidecar)                 │
        │  ├── test/           测试用微调数据集                       │
        │  │    ├── input_test_label_only_with_kg.json               │
        │  │    ├── input_test_reasoning_without_kg.json             │
        │  │    └── input_test_labels_clean.json (gold sidecar)      │
        │  └── aux/            辅助/临时产物                          │
        │       ├── pending_unmatched_pairs.json                     │
        │       └── regen_backup_20260426.json                       │
        └────────────────┬──────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     finetune_llama3_ddi.py   generate_outputs_distillation.py
     （LoRA 微调）              （外部 LLM 蒸馏生成 output）
              │                     │
              ▼                     │
   ┌────────────────────────┐       │
   │  results/              │       │
   │  llama3_ddi_lora_*/    │       │
   │  LoRA adapter 权重     │       │
   │  【整个 results/       │       │
   │   都 gitignore】       │       │
   └────────────┬───────────┘       │
                │                   │
                └────────┬──────────┘
                         │
                 inference_llama3_ddi.py
                         │
                         ▼
        ┌───────────────────────────────────────────────────────────┐
        │  data/reports/{experiment_name}/     [追踪]                │
        │  ├── inference/                                            │
        │  │    ├── test_out_*.json           推理原始输出            │
        │  │    └── test_metrics_*.json       分类指标                │
        │  ├── explanation_eval/                                     │
        │  │    ├── faithfulness_detail_*.json                       │
        │  │    └── faithfulness_summary_*.json                      │
        │  ├── judge_eval/                                           │
        │  │    ├── judge_detail_*.json                              │
        │  │    └── judge_summary_*.json                             │
        │  ├── config_snapshot.json                                  │
        │  ├── run_summary.json                                      │
        │  ├── compare_summary.csv                                   │
        │  └── artifact_validation.json                              │
        └────────────────────────────────────────────────────────────┘
```

---

## 2. 每层职责与命名规范

### `data/raw/`（追踪）
- **谁产出**：DDIcorpus 原始下载文件，手工放入。
- **谁消费**：`data_process/extract_*_labels.py`、`data_process/data_augmentation.py`。
- **命名**：保持原始文件名（`rbert_train.csv`、`rbert_test.csv` 等），不改动。
- **变更频率**：几乎不变。

### `data/labels/`（追踪）
- **谁产出**：`data_process/extract_train_labels.py`、`data_process/extract_test_labels.py`。
- **谁消费**：`data_process/build_finetune_dataset.py`。
- **命名**：`{label}.json` 与 `{label}_test.json`，`label ∈ {mechanism, effect, advise, int, false}`。
- **变更频率**：低（源 CSV 或抽取逻辑变化时才重跑）。

### `data/kg/`（大文件不追踪，小文件追踪）
- **谁产出**：外部 PrimeKG（`kg.csv`）；`data_process/build_prime_kg.py`（`primekg_graph.pkl` / `primekg_pyg.pt`）。
- **谁消费**：`prime_kg_utils/*`、`data_process/build_finetune_dataset.py`。
- **命名固定**：`kg.csv`、`primekg_graph.pkl`、`primekg_pyg.pt`、`drug_name_map.json`。
- **git 策略**：只有 `drug_name_map.json` 追踪；`*.csv/*.pkl/*.pt` 全部 gitignore（体积过大）。

### `data/finetune/{train,test,aux}/`（追踪）
- **谁产出**：`data_process/build_finetune_dataset.py`；蒸馏阶段（`generate_outputs_distillation.py`）会往同一份文件里 merge output 字段。
- **谁消费**：`finetune_llama3_ddi.py`、`inference_llama3_ddi.py`。
- **命名规范**：
  - 训练输入：`input.json`（`ablation_mode=full`）或 `input_<ablation_mode>.json`
  - 测试输入：`input_test.json` 或 `input_test_<ablation_mode>.json`
  - Sidecar 标签：与主文件同目录，`input(_test)_labels.json` 或 `input_test_labels_clean.json`
- **变更频率**：中（KG 或消融策略调整时重跑）。

### `data/reports/{experiment_name}/`（追踪）
- **谁产出**：`inference_llama3_ddi.py`、`evaluate/*`、`pipelines/experiment_pipeline.py`。
- **子结构固定**：
  - `inference/` — 推理原始输出（`test_out_*.json`）+ 分类指标（`test_metrics_*.json`）
  - `explanation_eval/` — faithfulness 事实度评估产物
  - `judge_eval/` — LLM-as-judge 打分产物
  - 顶层：`config_snapshot.json`、`run_summary.json`、`compare_summary.csv`、`artifact_validation.json`
- **git 策略**：全部追踪（这是**研究的关键产物**，需要能被 review 与横向对比）。

### `results/`（**整体 gitignore**）
- **只放 LoRA adapter 权重**（每个 adapter 目录几百 MB）。
- 目录约定：`results/llama3_ddi_lora_<mode>/`（保持与既有实验机上的路径一致）。
- 实验机上会有权重；GitHub 上永远看不到 `results/` 里的任何东西。

---

## 3. git 追踪对照表

| 路径 | git 状态 | 说明 |
|---|---|---|
| `data/raw/*.csv` | ✅ 追踪 | DDIcorpus 源数据，<20MB |
| `data/labels/*.json` | ✅ 追踪 | 抽取后的五类标签 |
| `data/kg/kg.csv` | ❌ gitignore | 数百 MB |
| `data/kg/primekg_graph.pkl` | ❌ gitignore | 数百 MB |
| `data/kg/primekg_pyg.pt` | ❌ gitignore | 数百 MB |
| `data/kg/drug_name_map.json` | ✅ 追踪 | 15 KB，药名对齐可复现 |
| `data/finetune/**/*.json` | ✅ 追踪 | ~50MB，加速实验机 clone 后即可训练 |
| `data/reports/**/*` | ✅ 追踪 | 评估报告，核心研究产物 |
| `results/**/*` | ❌ gitignore | 只放 LoRA 权重 |
| `models/**/*` | ❌ gitignore | 基座大模型 |
| `__pycache__/`、`*.pyc` | ❌ gitignore | Python 字节码 |
| `.idea/`、`.vscode/`、`.DS_Store` | ❌ gitignore | 编辑器/系统 |
| `.git/credentials` | ❌ gitignore | 本地 GitHub PAT |

---

## 4. data_process 各脚本一览

所有路径默认值都从 `data_process/paths.py` 常量读取，命令行覆盖优先。

### 4.1 `extract_train_labels.py`
从 `data/raw/standardized_generated_rbert_train_random.csv` 提取五类标签，产出到 `data/labels/{label}.json`。

```bash
python -m data_process.extract_train_labels
# 或
python data_process/extract_train_labels.py
```

### 4.2 `extract_test_labels.py`
从 `data/raw/rbert_test.csv` 提取五类测试标签，产出到 `data/labels/{label}_test.json`。

```bash
python -m data_process.extract_test_labels
```

### 4.3 `data_augmentation.py`
从 `data/raw/rbert_test.csv` 增广得到 `data/raw/test_augmented_data.csv`，用于分类评估。

```bash
python -m data_process.data_augmentation
```

### 4.4 `build_prime_kg.py`
从 `data/kg/kg.csv` 构建 `data/kg/primekg_graph.pkl`（查询用）与可选 `primekg_pyg.pt`（PyG 图）。

```bash
python -m data_process.build_prime_kg
```

### 4.5 `build_finetune_dataset.py`
读 `data/labels/*.json` + `data/kg/primekg_graph.pkl`，产出 `data/finetune/{train,test}/input(_test)_<mode>.json`。

```bash
# 训练集 (label_only_with_kg 模式)
python -m data_process.build_finetune_dataset \
    --ablation_mode label_only_with_kg \
    --use_all_data

# 测试集
python -m data_process.build_finetune_dataset \
    --ablation_mode label_only_with_kg \
    --input_suffix _test \
    --use_all_data
```

### 4.6 `generate_outputs_distillation.py`
调用外部 LLM（qwen-plus / qwen-max）为 `data/finetune/{train,test}/input*.json` 生成蒸馏 output 字段。

```bash
python data_process/generate_outputs_distillation.py \
    --config configs/experiments/restart_explanation_bootstrap.yaml \
    --config_section distillation_generation
```

### 4.7 `apps/run_external_direct_eval.py`（外部大模型直接评估支线）
**跳过训练与 LoRA 推理**，直接让外部 LLM 对 test 集生成 output 并评估，作为微调后模型的对照基线。可选启用 Agent Loop 反思-重试。

```bash
export DASHSCOPE_API_KEY=sk-xxx
python -m apps.run_external_direct_eval \
    --config configs/experiments/external_direct_eval_baseline.yaml
```

产出目录结构（与 LoRA 微调流程一致，方便横向对比）：

```text
data/reports/{external_direct_eval_experiment}/
├── config_snapshot.json
├── run_summary.json
├── inference/
│   ├── test_out.json                    ← 外部 LLM 产出的解释（+ agent_loop 元信息）
│   └── agent_trace_<run_id>.json        ← agent loop 每轮反思轨迹（启用时）
├── explanation_eval/
│   ├── faithfulness_summary_*.json
│   └── faithfulness_detail_*.json
└── judge_eval/
    ├── judge_summary_*.json
    └── judge_detail_*.json
```

Agent Loop 机制详见 [`agent_loop.md`](agent_loop.md)。

---

## 5. 实验机首次同步流程

**前提**：实验机上已有：
- 基座模型（如 `models/Meta-Llama-3-8B-Instruct/`）
- PrimeKG 原始 `kg.csv`（未上传 GitHub，需手工放到 `data/kg/kg.csv`）
- 之前训练好的 LoRA adapter（如 `results/llama3_ddi_lora_reasoning_without_kg/`），如果要复用

**步骤**：

```bash
# 1. Clone
git clone https://github.com/lufeiziaomodi/LLMAndPreModelForMed.git
cd LLMAndPreModelForMed
git checkout refactor/data-flow   # 切到重构分支（合并 main 前）

# 2. 恢复 gitignore 的大文件
mkdir -p data/kg models
# 把实验机上原有的 kg.csv 放到位
cp /some/where/kg.csv data/kg/kg.csv
# 把之前的 LoRA adapter 目录搬到新位置（保持 results/ 结构不变）
# 若之前已经在 results/llama3_ddi_lora_reasoning_without_kg/ 下，无需移动
# 若在旧的分散位置，用 mv 迁移

# 3. 一次性重建 KG 查询图
python -m data_process.build_prime_kg
# 产出 data/kg/primekg_graph.pkl

# 4. （可选）如果 data/finetune/ 下的 JSON 不满足需要，重新构造
# 训练 + 测试各跑一遍
python -m data_process.build_finetune_dataset \
    --ablation_mode label_only_with_kg --use_all_data
python -m data_process.build_finetune_dataset \
    --ablation_mode label_only_with_kg --input_suffix _test --use_all_data

# 5a. 完整链路：LoRA 微调 + 推理 + 评估（假设 adapter 已就位或让本次训练重新生成）
export DASHSCOPE_API_KEY=sk-xxx   # judge 阶段需要
python -m apps.run_experiment \
    --config configs/experiments/restart_explanation_bootstrap.yaml

# 5b. 支线：外部大模型"直接评估"（跳过训练，最快出结果，可选 agent loop）
python -m apps.run_external_direct_eval \
    --config configs/experiments/external_direct_eval_baseline.yaml

# 6. 评估报告落在 data/reports/{exp_name}/，
#    直接 git add / commit / push 即可与其他机器同步
```

---

## 6. 迁移前后路径对照表

| 旧路径 (main 分支) | 新路径 (refactor/data-flow 分支) |
|---|---|
| `data/rbert_train.csv` | `data/raw/rbert_train.csv` |
| `data/rbert_test.csv` | `data/raw/rbert_test.csv` |
| `data/standardized_generated_rbert_train_random.csv` | `data/raw/standardized_generated_rbert_train_random.csv` |
| `data/test_augmented_data.csv` | `data/raw/test_augmented_data.csv` |
| `data/mechanism.json` (及其他 4 类 + `_test`) | `data/labels/mechanism.json` (及其他) |
| `data/drug_name_map.json` | `data/kg/drug_name_map.json` |
| `data/kg.csv` (gitignored) | `data/kg/kg.csv` (gitignored) |
| `data/primekg_graph.pkl` (gitignored) | `data/kg/primekg_graph.pkl` (gitignored) |
| `data/primekg_pyg.pt` (gitignored) | `data/kg/primekg_pyg.pt` (gitignored) |
| `data/finetune_dataset_input_label_only_with_kg.json` | `data/finetune/train/input_label_only_with_kg.json` |
| `data/finetune_dataset_input_label_only_without_kg.json` | `data/finetune/train/input_label_only_without_kg.json` |
| `data/finetune_dataset_input_reasoning_without_kg.json` | `data/finetune/train/input_reasoning_without_kg.json` |
| `data/finetune_dataset_input_test_label_only_with_kg.json` | `data/finetune/test/input_test_label_only_with_kg.json` |
| `data/finetune_dataset_input_test_reasoning_without_kg.json` | `data/finetune/test/input_test_reasoning_without_kg.json` |
| `data/finetune_dataset_input_test_labels_clean.json` | `data/finetune/test/input_test_labels_clean.json` |
| `data/pending_unmatched_pairs.json` | `data/finetune/aux/pending_unmatched_pairs.json` |
| `results/eval_report/{name}/**` | `data/reports/{name}/**` |
| `results/test_predictions/*.json` | `data/reports/_legacy_test_predictions/*.json` |
| `results/experiments/default/eval_results.json` | `data/reports/_legacy_experiments/default_eval_results.json` |
| `results/llama3_ddi_lora_*/` | **保持不变**（LoRA adapter，`results/` 整体 gitignore） |
| `regen/` | 已删除（旧临时避让区，历史在 git 里可查） |
| `tmp_eval_out/` | 已删除（评估调试痕迹） |
| 根目录 `cimetidine_tree.json` | `data/reports/legacy_demos/cimetidine_tree.json` |
| 根目录 `rbert_test_predict.csv`、`test_predict.csv` | `data/reports/legacy_classifier/*` |
| 根目录 `finetune_dataset_input_label_only_with_kg_regen_20260426.json` | `data/finetune/aux/regen_backup_20260426.json` |

---

## 7. 添加新数据 / 新实验时的最佳实践

1. **新增源 CSV** → 放 `data/raw/`，如果大于 50MB 记得加 `.gitignore` 规则。
2. **新增标签抽取脚本** → 放 `data_process/`，输入用 `paths.DATA_RAW` 系列常量，输出到 `paths.DATA_LABELS`。
3. **新增微调数据集变体** → 用 `build_finetune_dataset.py --ablation_mode` 参数控制；不要手工在 `data/finetune/` 下创建目录。
4. **新增实验** → 复制 `configs/experiments/template_dual_track.yaml`，改 `experiment.name`；`output_root` 保持 `data/reports`。评估产物会自动落到 `data/reports/{new_name}/`。
5. **新增路径常量** → 只在 `data_process/paths.py` 里加，禁止在其它文件写字符串字面量。

---

## 8. 常见错误

| 错误 | 原因 | 修正 |
|---|---|---|
| `FileNotFoundError: data/labels/mechanism.json` | `extract_train_labels.py` 没跑过 | `python -m data_process.extract_train_labels` |
| `FileNotFoundError: data/kg/primekg_graph.pkl` | KG 图缓存没构建 | `python -m data_process.build_prime_kg` |
| pipeline 报错 `results/eval_report/... not found` | 用了旧配置 | 更新 `experiment.output_root: data/reports` |
| judge 阶段报 401 / 匿名 API 拒绝 | `${DASHSCOPE_API_KEY}` 未 export | `export DASHSCOPE_API_KEY=sk-xxx` |
| LoRA 训练无法找到 base model | `models/` gitignore，实验机无权重 | 手工把权重放到 `models/Meta-Llama-3-8B-Instruct/` |
