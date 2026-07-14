import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_test_data(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            label = str(row[0]).strip().lower()
            text = str(row[1]).strip()
            if not text:
                continue
            rows.append((label, text))
    return rows


def load_predictions_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Predictions JSON must be a list")
    return data


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
