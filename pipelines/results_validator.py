from pathlib import Path
from typing import Any, Dict, List


def _check_paths(paths: List[str]) -> Dict[str, Any]:
    checked = []
    missing = []
    for p in paths:
        if not p:
            continue
        exists = Path(p).exists()
        checked.append({"path": p, "exists": exists})
        if not exists:
            missing.append(p)
    return {
        "checked": checked,
        "missing": missing,
        "ok": len(missing) == 0,
    }


def validate_experiment_artifacts(summary: Dict[str, Any]) -> Dict[str, Any]:
    stages = summary.get("stages", {})
    all_paths: List[str] = []

    cls = stages.get("classification_eval", {})
    all_paths.extend(cls.get("artifacts", []))

    exp = stages.get("explanation_eval", {})
    all_paths.extend(exp.get("artifacts", []))

    check = _check_paths(all_paths)
    return {
        "n_artifacts": len(all_paths),
        "ok": check["ok"],
        "missing": check["missing"],
        "checked": check["checked"],
    }
