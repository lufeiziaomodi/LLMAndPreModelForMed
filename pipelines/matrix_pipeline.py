import copy
import os
from typing import Any, Dict, List

from pipelines.experiment_pipeline import execute_experiment
from pipelines.io_utils import save_json


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def run_matrix(conf: Dict[str, Any]) -> Dict[str, Any]:
    matrix = conf.get("matrix", {})
    variants: List[Dict[str, Any]] = matrix.get("variants", [])
    if not variants:
        raise ValueError("matrix.variants is required for matrix run")

    results: List[Dict[str, Any]] = []
    for idx, variant in enumerate(variants, 1):
        name = variant.get("name", f"variant_{idx}")
        patch = variant.get("patch", {})
        merged = _deep_update(conf, patch)
        merged.setdefault("experiment", {})
        merged["experiment"]["name"] = str(merged["experiment"].get("name", "exp")) + f"__{name}"
        print(f"[Matrix] Running {idx}/{len(variants)}: {merged['experiment']['name']}")
        result = execute_experiment(merged)
        results.append(
            {
                "name": name,
                "run_id": result.get("run_id"),
                "output_dir": result.get("output_dir"),
                "compare_summary_csv": result.get("compare_summary_csv", ""),
            }
        )

    root_output = conf.get("experiment", {}).get("output_root", "data/reports")
    os.makedirs(root_output, exist_ok=True)
    out_path = os.path.join(root_output, "matrix_runs_summary.json")
    save_json(out_path, {"runs": results})
    return {"runs": results, "summary_path": out_path}
