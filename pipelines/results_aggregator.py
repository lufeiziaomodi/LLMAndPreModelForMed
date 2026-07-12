import csv
from pathlib import Path
from typing import Any, Dict, List


def collect_compare_rows(root_dir: str) -> List[Dict[str, Any]]:
    root = Path(root_dir)
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("compare_summary.csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source_file"] = str(path)
                rows.append(dict(row))
    return rows


def write_aggregate_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out, "w", encoding="utf-8") as f:
            f.write("experiment_name,model_name,track,micro_f1,macro_f1,weighted_f1,coverage_mean,hallucination_mean,consistency_mean,mechanism_overall_score_mean,source_file\n")
        return str(out)

    fieldnames = [
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
        "source_file",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return str(out)
