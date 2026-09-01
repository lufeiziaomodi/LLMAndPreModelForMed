# 论文创新点、消融实验矩阵与双机协作规范

更新时间：2026-09-01

## 1. 论文的核心研究问题

本文不只研究“模型能否识别 DDI 标签”，而是研究以下问题：

1. 如何从临床句子与知识图谱中提取与当前 DDI 现象真正相关的机制证据？
2. 如何同时评价解释的事实可靠性与机制解释价值？
3. 如何让高成本外部模型只处理本地模型的困难样本，并将修复结果转化为后续训练数据？

建议论文主线表述为：

> 面向药物相互作用机制解释，构建句子条件的知识图谱证据筛选、事实—机制双通道评价，以及本地模型主推理与外部模型纠错相结合的闭环蒸馏框架。

## 2. 创新点与验证关系

| 创新点 | 核心设计 | 解决的问题 | 必须提供的实验依据 | 当前成熟度 |
|---|---|---|---|---|
| 句子条件的 KG 机制证据压缩 | 先识别 metabolism、absorption、renal clearance、QT 等句子 cue，再筛选 CYP、UGT、SLC、ABC、离子通道等机制锚点，最后生成 Evidence Summary | 原始一阶邻居噪声大，节点关联不等于因果关系 | No-KG、Raw-KG、规则过滤 KG、句子条件 Filtered-KG 四级消融 | No-KG/Raw-KG 全量对照已完成；过滤与条件化待实验 |
| 现象级结构化机制解释 | 用 `query_group` 表达同一现象涉及的药物集合；输出 `GLOBAL_PHENOMENON`、短机制链和最多 3 个代表药物对 | 逐 pair 枚举重复、token 浪费、解释割裂 | 逐 pair、分组但不压缩、分组 + 最多 3 对三组消融 | 输出协议已在外部直评入口落地 |
| 事实—机制解耦的双通道评价 | Faithfulness 判断 claim 是否有证据；Judge 独立评价链条完整性、方向、粒度、一致性、不确定性和临床可用性 | 单一指标无法区分“保守复述”和“丰富但无证据” | 与人工标注的相关性、错误接受率及 Faith-only/Judge-only/Dual-gate 消融 | 代码已落地，人工校准待做 |
| 成本感知的双模型修复闭环 | 本地 LoRA 批量推理；重复失败时由外部模型读取原推理链和评价缺口重新推理；通过质量门后进入增量蒸馏集 | 外部模型全量调用成本高，本地模型困难样本又会重复失败 | Single Pass、无反馈外部重写、Faith 反馈修复、Faith + Judge 修复、增量微调五级消融 | 基础同模型 Agent Loop 已实现，跨模型救援待完成 |
| 分类与解释双轨学习 | 分类模型只输出五分类；解释模型独立学习机制链，使用不同训练数据和评价指标 | 标签正确不代表机制解释正确 | 分类轨道 With/No-KG；解释轨道 Base/LoRA/Direct/Agent 对照 | 工程链路已建立 |

其中前三项可以作为论文的主要方法创新；双模型闭环是系统创新，也是后续实验的重点。双轨学习更适合作为总体框架设计，不宜单独夸大为算法创新。

## 3. 核心主结果表

论文主结果建议拆成三张表，避免把不同量纲混在一起。

### 表 1：分类轨道结果

| 模型 | KG 输入 | 样本数 | Accuracy | Macro-F1 | Weighted-F1 | 备注 |
|---|---|---:|---:|---:|---:|---|
| Llama3 LoRA | No-KG | 1042 | 待统一重跑 | 待统一重跑 | 待统一重跑 | 主分类基线 |
| Llama3 LoRA | Raw-KG | 1042 | 待统一重跑 | 待统一重跑 | 待统一重跑 | 检验原始 KG 是否带来增益 |
| Llama3 LoRA | Filtered-KG | 1042 | 可选 | 可选 | 可选 | 仅在分类轨道也使用筛选证据时加入 |

### 表 2：解释质量主结果

| 编号 | 模型 | 训练方式 | 推理证据 | Agent | Coverage ↑ | Hallucination ↓ | Query Coverage ↑ | KG Grounded ↑ | Mechanism Score ↑ | Good Rate ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| M1 | Llama3 Base | 无微调 | No-KG | 关闭 | 待统一重评 | 待统一重评 | 待统一重评 | — | 待统一重评 | 待统一重评 |
| M2 | Llama3 LoRA | No-KG 蒸馏 | No-KG | 关闭 | 待统一重评 | 待统一重评 | 待统一重评 | — | 待统一重评 | 待统一重评 |
| M3 | Qwen-Plus | Direct | No-KG | 关闭 | 0.0291 | 0.4548 | 0.8732 | — | 8.4904 | 0.2831 |
| M4 | Qwen-Plus | Direct | Raw-KG | 关闭 | 0.0128 | 0.3255 | 0.8764 | 0.3714 | 8.5938† | 0.2964† |
| M5 | Qwen-Plus | Direct | Filtered-KG | 关闭 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| M6 | Llama3 LoRA | Filtered-KG 蒸馏 | Filtered-KG | 关闭 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| M7 | Llama3 LoRA + External Repair | Filtered-KG 蒸馏 | Filtered-KG | 开启 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

`†` Raw-KG Judge 在 1042 条中有 1039 条有效结果；3 条超长 KG 请求失败并单独报告，不计入有效均值。No-KG 与 Raw-KG 的 Faithfulness 均使用 2026-09-01 校准后的相同口径。

阶段性解释：Raw-KG 明显降低 unsupported claim 比例，但主要将 claim 推向 partial support，而非 strict support；同时 3 条最长证据触发 qwen-max 上下文拒绝。因此下一步不是继续扩大邻居，而是验证 Anchor Filter 与句子 cue 条件化能否把“更多锚点”转化为“更可靠的因果链”。

### 表 3：Agent Loop 质量—成本结果

| 设置 | 首轮通过率 ↑ | 最终通过率 ↑ | 平均轮次 ↓ | 最大轮次失败率 ↓ | 外部调用/样本 ↓ | 延迟/样本 ↓ | Mechanism Score ↑ | Hallucination ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 本地 Single Pass | 待运行 | 同首轮 | 1.00 | — | 0 | 待测 | 待测 | 待测 |
| 本地失败后外部重写，无评价反馈 | 待运行 | 待运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 外部修复 + Faithfulness Critic | 待运行 | 待运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 外部修复 + Judge Critic | 待运行 | 待运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 外部修复 + Dual Critic | 待运行 | 待运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 4. 消融实验表

### 消融 A：KG 证据构造

保持模型、测试样本、生成温度和输出协议不变，只改变证据输入。

| 组别 | 一阶邻居 | 可靠锚点过滤 | 句子 cue 条件化 | Evidence Summary | 目的 |
|---|---:|---:|---:|---:|---|
| A0 No-KG | 否 | 否 | 否 | 否 | 无知识基线 |
| A1 Raw-KG | 是 | 否 | 否 | 否 | 检验直接拼接节点是否引入噪声 |
| A2 Anchor Filter | 是 | 是 | 否 | 否 | 隔离低质量 PPI/泛化节点过滤的贡献 |
| A3 Cue-conditioned KG | 是 | 是 | 是 | 否 | 检验句子语义是否提高证据相关性 |
| A4 Full Evidence Summary | 是 | 是 | 是 | 是 | 完整方法 |

主指标：Coverage、Hallucination、KG Grounded、机制粒度、Judge 总分、输入 token 数。若 A1 不优于 A0，而 A3/A4 显著提升，可支撑“关键不是更多 KG，而是更相关的 KG”。

### 消融 B：输出结构

| 组别 | query 表达 | 输出粒度 | 代表药物对限制 | 目的 |
|---|---|---|---:|---|
| B0 | 逐 pair 列表 | 每 pair 一段解释 | 无 | 传统输出基线 |
| B1 | `query_group` | 每 pair 一段解释 | 无 | 单独检验输入分组作用 |
| B2 | `query_group` | 现象级解释 | 无 | 检验现象级归纳作用 |
| B3 | `query_group` | 现象级解释 | 最多 3 对 | 完整结构化输出 |

主指标：Query Coverage、重复 claim 比例、输出 token 数、单样本解析成功率、机制链完整性，以及人工可读性评分。

### 消融 C：蒸馏与知识迁移

| 组别 | 基座 | 蒸馏数据 | 推理 KG | 目的 |
|---|---|---|---|---|
| C0 | Llama3 Base | 无 | No-KG | 基座下限 |
| C1 | Llama3 LoRA | No-KG reasoning | No-KG | 蒸馏本身的贡献 |
| C2 | Llama3 LoRA | Raw-KG reasoning | Raw-KG | 原始 KG 蒸馏基线 |
| C3 | Llama3 LoRA | Filtered-KG reasoning | Filtered-KG | 完整知识增强蒸馏 |
| C4 | Qwen-Plus Direct | 无 | Filtered-KG | 教师模型上界参照 |

LoRA 至少使用 3 个随机种子，报告均值、标准差和 95% bootstrap 置信区间。

### 消融 D：评价体系

| 组别 | Faithfulness | Judge | 准入规则 | 目的 |
|---|---:|---:|---|---|
| D0 | 否 | 否 | 无自动门控 | 原始生成结果 |
| D1 | 是 | 否 | 只看证据支持 | 是否错误接收“保守复述” |
| D2 | 否 | 是 | 只看机制得分 | 是否错误接收“丰富但脱离证据” |
| D3 | 是 | 是 | 双门槛 | 完整评价体系 |

用 100–200 条人工双人标注样本校准，报告 Spearman 相关系数、Cohen's kappa、错误接受率和错误拒绝率。

### 消融 E：Agent Loop

| 组别 | 初始生成 | 外部修复 | 反馈内容 | 是否回流增量集 | 目的 |
|---|---|---:|---|---:|---|
| E0 | 本地 LoRA | 否 | 无 | 否 | Single Pass 基线 |
| E1 | 本地 LoRA | 是 | 只提供原推理链 | 否 | 外部模型能力贡献 |
| E2 | 本地 LoRA | 是 | Faithfulness gaps | 否 | 事实反馈贡献 |
| E3 | 本地 LoRA | 是 | Judge gaps | 否 | 机制反馈贡献 |
| E4 | 本地 LoRA | 是 | Faithfulness + Judge gaps | 否 | 完整在线修复 |
| E5 | 增量微调后的 LoRA | 按需 | 双反馈 | 是 | 闭环学习是否减少后续外部调用 |

除质量指标外，必须报告触发率、平均外部调用数、平均延迟、API token/成本和增量微调后外部救援率的下降幅度。

## 5. 统计与公平性要求

1. 所有核心模型使用同一 1042 条测试样本；无法生成的样本保留并计为失败，不得静默删除。
2. No-KG、Raw-KG、Filtered-KG 使用相同模型版本、temperature、max tokens 和输出 schema。
3. LoRA 实验至少 3 个随机种子；API Direct 在 temperature=0 下运行一次全量，并保留失败重试记录。
4. 样本级指标使用配对 bootstrap 计算 95% CI；关键方法比较同时报告效应量，而不只报告 p 值。
5. 人工评价者在不知道模型名称和实验组别的条件下盲评。
6. Mock 数值只能用于实验前设定验收区间，不能进入论文主结果表。

## 6. 两台机器的职责划分

| 内容 | 笔记本 | 实验机器 | GitHub 是否同步 |
|---|---|---|---:|
| 文档、论文、实验设计 | 主负责 | 可审阅 | 是 |
| 配置、数据处理、评价代码 | 主负责 | 拉取后执行 | 是 |
| 外部 API Direct 实验 | 可执行 | 也可执行 | 配置与报告同步 |
| PrimeKG 图构建 | 不要求 | 主负责 | 仅脚本和小型映射同步 |
| Llama3/LoRA 训练推理 | 不执行或只 smoke | 主负责 | 权重不同步，报告同步 |
| `models/`、`results/`、大型 KG | 本地可无 | 本地完整保存 | 否 |
| `data/finetune/` 与 `data/reports/` | 拉取、分析 | 生成、提交 | 是 |
| API Key、PAT、`.env` | 仅本地环境变量 | 仅本地环境变量 | 否 |

## 7. GitHub 交接协议

当前仓库远端为 `origin`，当前工作分支为 `refactor/data-flow`。两台机器采用串行交接，避免同时向同一分支写入。

### 7.1 笔记本发起实验

1. 完成代码、配置和实验说明。
2. 确认 `git status`，只暂存本次需要同步的文件，禁止使用未经检查的 `git add .`。
3. 提交并推送，例如：

```powershell
git add configs/experiments/<config>.json docs/<plan>.md <changed-code-files>
git commit -m "exp: prepare E07 filtered KG direct evaluation"
git push origin refactor/data-flow
```

4. 在同一聊天窗口说明实验编号、提交 SHA、运行命令和预期产物目录。

### 7.2 实验机器接收并运行

```bash
git status
git fetch origin
git pull --ff-only origin refactor/data-flow
git rev-parse --short HEAD
```

运行前确认本地 `models/`、`results/`、`data/kg/` 仍然存在。Git pull 不会删除这些被忽略的文件。

实验结束后，只提交配置快照、训练/推理摘要和评估报告：

```bash
git add data/reports/<experiment_name>/ data/finetune/<new-small-dataset-files>
git commit -m "exp: complete E07 qwen-plus filtered KG evaluation"
git push origin refactor/data-flow
```

模型权重继续保存在实验机器的 `results/`，不提交。

### 7.3 笔记本回收结果

```powershell
git status
git pull --ff-only origin refactor/data-flow
```

然后检查：

- `config_snapshot.json` 是否记录实际运行配置；
- `run_summary.json` 中样本数和失败数；
- Faithfulness/Judge detail 与 summary 是否齐全；
- `compare_summary.csv` 是否可以加入论文表格；
- 是否存在空输出、样本丢失或实验配置漂移。

## 8. 每次实验的交接消息模板

```text
实验编号：E07
代码提交：<git SHA>
机器：实验机器 / 笔记本
配置：configs/experiments/<name>.json
运行命令：python -m ...
输入数据：data/finetune/...
预期样本数：1042
输出目录：data/reports/<experiment_name>/
需要同步：run_summary、compare_summary、inference、faithfulness、judge
本地保留但不上传：models、results、data/kg 大文件、API Key
```

同一时刻只让一台机器对 `refactor/data-flow` 分支产生新提交。若两台机器必须并行实验，应为每个实验建立独立分支，再由笔记本合并，避免报告与配置互相覆盖。
