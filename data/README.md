# `data/` 目录布局

数据分层，由 `data_process/` 驱动。详细流程见 [`docs/data_flow.md`](../docs/data_flow.md)。

```
data/
├── raw/          源 DDIcorpus CSV（追踪，<20MB）
├── labels/       五类标签 JSON（extract_*_labels.py 产出，追踪）
├── kg/           PrimeKG 资源（*.csv/*.pkl/*.pt 大文件不追踪）
├── finetune/     微调数据集 JSON
│   ├── train/    训练输入
│   ├── test/     测试输入 + gold labels sidecar
│   └── aux/      pending_unmatched_pairs / regen backup 等衍生
└── reports/      推理与评估产物（按 experiment_name 分子目录，追踪）
```

**规则**：
- 每个文件路径都对应 `data_process/paths.py` 里的常量，代码里禁止写字符串字面量拼接。
- `data/kg/kg.csv`、`data/kg/primekg_graph.pkl`、`data/kg/primekg_pyg.pt` 因为体积过大不进 git，实验机需要手工放置。
- `results/` 目录整体 gitignore，只放 LoRA adapter 权重。
