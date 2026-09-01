# 双机实验交接单

更新时间：2026-09-01

本文件是笔记本与实验机器之间的唯一当前状态。聊天记录用于解释背景，但代码版本、实验状态和下一条命令以本文件与 GitHub 为准。

## 当前交接状态

| 项目 | 当前值 |
|---|---|
| 当前执行方 | 实验机器（待接收） |
| 交接来源 | 笔记本 |
| Git 远端 | `origin` |
| 工作分支 | `refactor/data-flow` |
| 最低兼容提交 | `f2f17d0`（接收方应拉取该分支最新提交） |
| 当前实验阶段 | E5 No-KG 与 E6 Raw-KG 全量实验已完成 |
| 下一目标 | 构建 Filtered-KG / Evidence Summary 数据 |
| 暂不执行 | Agent Loop 全量实验、LoRA 增量微调 |

## 已完成并已同步

- E5：Qwen-Plus Direct No-KG，1042 条全量生成和评价。
- E6：Qwen-Plus Direct Raw-KG，1042 条生成全部成功；1039 条 qwen-max Judge 有效。
- 已确认 Raw-KG 能降低 unsupported claim，但会增加 partial support，并产生极端长上下文。
- 评价器已修复 CYP 编号切分、代表药物对覆盖率、KG anchor grounding、Judge 失败均值污染和配置快照泄密问题。
- 正式报告位于 `data/reports/external_direct_eval_qwen_plus_with_kg/`。

## 实验机器接收步骤

在实验机器的仓库根目录执行：

```powershell
git status --short
git fetch origin
git switch refactor/data-flow
git pull --ff-only origin refactor/data-flow
git rev-parse --short HEAD
powershell -ExecutionPolicy Bypass -File scripts/verify_handoff.ps1 -ExpectedCommit f2f17d0
```

验收条件：

- `HEAD` 为 `f2f17d0` 或其后续提交。
- 没有被追踪文件的未提交修改。
- 实验机器本地仍保留 PrimeKG、模型权重与训练结果。
- `DASHSCOPE_API_KEY` 可用；脚本只检查是否存在，不输出 Key。

如果 `git status` 显示本地已有修改，不要覆盖、重置或强制拉取。先让 Codex 判断这些修改是否属于未同步实验。

## 实验机器上的第一项工作

先做数据与依赖审计，不立即训练：

1. 定位实验机中完整 PrimeKG CSV、图缓存和药名映射。
2. 确认 `data_process/build_finetune_dataset.py` 当前能够读取哪些图格式。
3. 在 50 条固定样本上生成三组证据：Raw-KG、Anchor Filter、Cue-conditioned Evidence Summary。
4. 检查证据长度、有效 anchor 数、无证据比例和 5 个典型案例。
5. 小样本验收后，再生成 1042 条 Filtered-KG 测试集并提交小型数据产物与统计报告。

本阶段不更改外部生成模型。Filtered-KG Direct 仍使用 `qwen-plus` 生成与 `qwen-max` Judge，以保持 E5/E6/E7 可比。

## 回传规则

实验机器完成一个可验收阶段后：

1. 更新本文件的提交号、完成内容、失败信息和下一执行方。
2. 只提交代码、配置、小型数据、统计摘要和报告；不要提交 `models/`、`results/`、大型 KG 或任何 API Key。
3. 推送到 `origin/refactor/data-flow`。
4. 笔记本必须先执行 `git pull --ff-only` 和交接检查，再恢复写入。

两台机器禁止同时修改并推送 `refactor/data-flow`。当前交接完成前，笔记本只做阅读与讨论，不产生需要推送到该分支的新提交。

## 给实验机 Codex 的开场指令

```text
读取 docs/handoff_current.md、docs/data_flow.md、docs/agent_loop.md。
先运行 scripts/verify_handoff.ps1 -ExpectedCommit f2f17d0，审计本机被 .gitignore 忽略的 PrimeKG、模型和训练产物。
不要立即训练，也不要删除或覆盖本机文件。先汇报完整 KG 的实际路径、格式、规模，以及构建 50 条 Filtered-KG 探针所需的最小代码改动。
```
