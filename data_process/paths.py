"""统一的项目路径常量与目录布局。

任何脚本要引用 data/ 或 results/ 下的文件，都从这里 import，
禁止再散落字符串字面量。目录约定见 docs/data_flow.md。

分层：
    data/raw/            源数据 (rbert_train.csv/rbert_test.csv 等)
    data/labels/         extract_*_labels.py 产出的五类标签 json
    data/kg/             PrimeKG 资源 (kg.csv/primekg_graph.pkl/drug_name_map.json)
    data/finetune/train/ build_finetune_dataset.py 训练输入
    data/finetune/test/  build_finetune_dataset.py 测试输入 (+ labels sidecar)
    data/finetune/aux/   pending_unmatched_pairs / regen backup 等衍生副产物
    data/reports/{exp}/  推理与评估产物 (被 git 追踪，机器上可 pull 直接看)
    results/adapters/    LoRA 权重目录 (整个 results/ 被 gitignore)
"""

from __future__ import annotations

from pathlib import Path

# ---- 基础锚点 -------------------------------------------------------------
# data_process/paths.py -> data_process/ -> <项目根>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---- data/ 分层 -----------------------------------------------------------
DATA_ROOT: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_ROOT / "raw"
DATA_LABELS: Path = DATA_ROOT / "labels"
DATA_KG: Path = DATA_ROOT / "kg"
DATA_FINETUNE: Path = DATA_ROOT / "finetune"
DATA_FINETUNE_TRAIN: Path = DATA_FINETUNE / "train"
DATA_FINETUNE_TEST: Path = DATA_FINETUNE / "test"
DATA_FINETUNE_AUX: Path = DATA_FINETUNE / "aux"
DATA_REPORTS: Path = DATA_ROOT / "reports"

# ---- 常用具名文件（避免在多处重复拼接） -----------------------------------
# raw
RBERT_TRAIN_CSV: Path = DATA_RAW / "rbert_train.csv"
RBERT_TEST_CSV: Path = DATA_RAW / "rbert_test.csv"
STANDARDIZED_TRAIN_CSV: Path = DATA_RAW / "standardized_generated_rbert_train_random.csv"
TEST_AUGMENTED_CSV: Path = DATA_RAW / "test_augmented_data.csv"

# kg
KG_CSV: Path = DATA_KG / "kg.csv"
PRIMEKG_GRAPH_PKL: Path = DATA_KG / "primekg_graph.pkl"
PRIMEKG_PYG_PT: Path = DATA_KG / "primekg_pyg.pt"
DRUG_NAME_MAP_JSON: Path = DATA_KG / "drug_name_map.json"

# finetune aux
PENDING_UNMATCHED_JSON: Path = DATA_FINETUNE_AUX / "pending_unmatched_pairs.json"

# ---- results/ (只放 gitignored 权重) --------------------------------------
RESULTS_ROOT: Path = PROJECT_ROOT / "results"
# 兼容当前实验机上已存在的目录名，adapter 路径保留 results/llama3_ddi_lora_* 结构
LORA_REASONING_WITHOUT_KG_DIR: Path = RESULTS_ROOT / "llama3_ddi_lora_reasoning_without_kg"
LORA_LABEL_ONLY_WITH_KG_DIR: Path = RESULTS_ROOT / "llama3_ddi_lora_label_only_with_kg"


# ---- helper --------------------------------------------------------------
def report_dir(experiment_name: str) -> Path:
    """返回 data/reports/{experiment_name}/，用于统一收敛评估产物落盘位置。"""
    return DATA_REPORTS / experiment_name


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回，方便一句话完成 mkdir。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "DATA_RAW",
    "DATA_LABELS",
    "DATA_KG",
    "DATA_FINETUNE",
    "DATA_FINETUNE_TRAIN",
    "DATA_FINETUNE_TEST",
    "DATA_FINETUNE_AUX",
    "DATA_REPORTS",
    "RESULTS_ROOT",
    "RBERT_TRAIN_CSV",
    "RBERT_TEST_CSV",
    "STANDARDIZED_TRAIN_CSV",
    "TEST_AUGMENTED_CSV",
    "KG_CSV",
    "PRIMEKG_GRAPH_PKL",
    "PRIMEKG_PYG_PT",
    "DRUG_NAME_MAP_JSON",
    "PENDING_UNMATCHED_JSON",
    "LORA_REASONING_WITHOUT_KG_DIR",
    "LORA_LABEL_ONLY_WITH_KG_DIR",
    "report_dir",
    "ensure_dir",
]
