import os
from typing import Any, Dict, List

from pipelines.classification_pipeline import run_classification_eval
from pipelines.compare_pipeline import write_compare_summary
from pipelines.explanation_pipeline import run_explanation_eval
from pipelines.io_utils import make_run_id, save_json
from pipelines.inference_pipeline import run_inference
from pipelines.results_validator import validate_experiment_artifacts
from pipelines.training_pipeline import run_training


def execute_experiment(conf: Dict[str, Any]) -> Dict[str, Any]:
    exp = conf.get("experiment", {})
    exp_name = exp.get("name", "exp")
    run_id = make_run_id(exp_name)

    root_output = exp.get("output_root", "data/reports")
    output_dir = os.path.join(root_output, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "experiment": exp,
        "stages": {},
    }
    save_json(os.path.join(output_dir, "config_snapshot.json"), conf)

    training_conf = conf.get("training", {})
    if training_conf.get("enabled", False):
        summary["stages"]["training"] = run_training(training_conf)

    inference_conf = conf.get("inference", {})
    if inference_conf.get("enabled", False):
        inference_conf = dict(inference_conf)
        if not inference_conf.get("output_dir"):
            inference_conf["output_dir"] = os.path.join(output_dir, "inference")
        summary["stages"]["inference"] = run_inference(inference_conf)

    cls_conf = conf.get("classification_eval", {})
    if cls_conf.get("enabled", False):
        cls_conf = dict(cls_conf)
        cls_conf.setdefault("output_dir", os.path.join(output_dir, "classification_eval"))
        summary["stages"]["classification_eval"] = run_classification_eval(cls_conf)

    exp_conf = conf.get("explanation_eval", {})
    if exp_conf.get("enabled", False):
        exp_conf = dict(exp_conf)
        exp_conf.setdefault("output_dir", os.path.join(output_dir, "explanation_eval"))
        if not exp_conf.get("predictions_file") and "inference" in summary["stages"]:
            exp_conf["predictions_file"] = summary["stages"]["inference"].get("output_json", "")
        judge_conf = dict(conf.get("judge", {}))
        judge_conf.setdefault("output_dir", os.path.join(output_dir, "judge_eval"))
        summary["stages"]["explanation_eval"] = run_explanation_eval(exp_conf, judge_conf)

    compare_rows: List[Dict[str, Any]] = []
    if "classification_eval" in summary["stages"]:
        metrics = summary["stages"]["classification_eval"].get("metrics", {})
        compare_rows.append(
            {
                "experiment_name": exp.get("name", run_id),
                "model_name": conf.get("classification_eval", {}).get("model_name", "classification_model"),
                "track": "classification",
                "micro_f1": metrics.get("micro_f1", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
            }
        )

    if "explanation_eval" in summary["stages"]:
        faith = summary["stages"]["explanation_eval"].get("faithfulness", {})
        judge = summary["stages"]["explanation_eval"].get("judge", {})
        compare_rows.append(
            {
                "experiment_name": exp.get("name", run_id),
                "model_name": conf.get("explanation_eval", {}).get("model_name", "explanation_model"),
                "track": "explanation",
                "coverage_mean": faith.get("coverage_mean", ""),
                "hallucination_mean": faith.get("hallucination_mean", ""),
                "consistency_mean": faith.get("consistency_mean", ""),
                "mechanism_overall_score_mean": judge.get("mechanism_overall_score_mean", ""),
            }
        )

    if compare_rows:
        compare_csv = write_compare_summary(os.path.join(output_dir, "compare_summary.csv"), compare_rows)
        summary["compare_summary_csv"] = compare_csv

    validation = validate_experiment_artifacts(summary)
    summary["artifact_validation"] = validation
    save_json(os.path.join(output_dir, "artifact_validation.json"), validation)

    save_json(os.path.join(output_dir, "run_summary.json"), summary)
    summary["output_dir"] = output_dir
    return summary
