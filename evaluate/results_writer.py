import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_predictions_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "true_label",
            "predicted_label",
            "text",
            "reasoning",
            "coverage_ratio",
            "hallucination_rate",
            "consistency_score",
            "judge_overall",
        ])
        for r in rows:
            writer.writerow([
                r.get("true_label", ""),
                r.get("predicted_label", ""),
                r.get("text", ""),
                r.get("reasoning", ""),
                r.get("coverage_ratio", ""),
                r.get("hallucination_rate", ""),
                r.get("consistency_score", ""),
                r.get("judge_overall", ""),
            ])
