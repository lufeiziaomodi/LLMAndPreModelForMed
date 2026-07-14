import os
from typing import Any, Dict

from evaluate.cli.workflows import run_faithfulness, run_judge
from evaluate.config import EvaluationConfig
from pipelines.contracts import ExplanationEvalConfig, JudgeConfig


def run_explanation_eval(section: Dict[str, Any], judge_section: Dict[str, Any]) -> Dict[str, Any]:
    exp_conf = ExplanationEvalConfig.from_section(section)
    predictions_file = exp_conf.predictions_file
    if not predictions_file:
        raise ValueError("explanation_eval.predictions_file is required")

    cfg = EvaluationConfig(
        mode="faithfulness",
        predictions_file=predictions_file,
        predictions_labels_file=exp_conf.labels_file or None,
        label_predictions_file=exp_conf.label_predictions_file or None,
        output_dir=exp_conf.output_dir,
    )
    cfg.validate()

    faith = run_faithfulness(cfg)

    payload: Dict[str, Any] = {
        "faithfulness": faith["summary"],
        "tag": faith["tag"],
        "output_dir": cfg.output_dir,
        "artifacts": list(faith.get("artifacts", [])),
    }

    judge_conf = JudgeConfig.from_section(judge_section, default_output_dir=cfg.output_dir)
    if judge_conf.enabled:
        api_key = judge_conf.api_key or os.getenv("DASHSCOPE_API_KEY")
        judge_cfg = EvaluationConfig(
            mode="judge",
            predictions_file=predictions_file,
            predictions_labels_file=exp_conf.labels_file or None,
            label_predictions_file=exp_conf.label_predictions_file or None,
            output_dir=judge_conf.output_dir,
            use_judge=True,
            judge_model_id=judge_conf.model_id,
            judge_api_key=api_key,
            judge_base_url=judge_conf.base_url,
            judge_max_retries=judge_conf.max_retries,
            judge_retry_delay=judge_conf.retry_delay,
            judge_verbose=judge_conf.verbose,
            judge_checkpoint_every=judge_conf.checkpoint_every,
        )
        judge_cfg.validate()
        judge = run_judge(judge_cfg, rows=faith["rows"])
        payload["judge"] = judge["summary"]
        payload["artifacts"].extend(judge.get("artifacts", []))

    return payload
