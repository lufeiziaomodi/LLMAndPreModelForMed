from typing import Any, Dict

from evaluate.cli.workflows import run_classify
from evaluate.config import EvaluationConfig
from pipelines.contracts import ClassificationEvalConfig


def run_classification_eval(section: Dict[str, Any]) -> Dict[str, Any]:
    conf = ClassificationEvalConfig.from_section(section)
    cfg = EvaluationConfig(
        mode="classify",
        test_data_path=conf.test_data,
        output_dir=conf.output_dir,
        model_id=conf.model_id,
        max_new_tokens=conf.max_new_tokens,
    )
    cfg.validate()
    result = run_classify(cfg)
    return {
        "metrics": result["metrics"],
        "tag": result["tag"],
        "output_dir": cfg.output_dir,
        "artifacts": result.get("artifacts", []),
    }
