import csv
from pathlib import Path
from typing import Any, Dict, List


def write_compare_summary(output_csv: str, rows: List[Dict[str, Any]]) -> str:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "experiment_name",
        "model_name",
        "track",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "coverage_mean",
        "hallucination_mean",
        "consistency_mean",
        "mechanism_overall_score_mean",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return str(path)
