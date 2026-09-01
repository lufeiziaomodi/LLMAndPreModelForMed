"""外部大模型"直接评估"实验入口（跳过微调 / LoRA 推理）。

工程动机：
  在正式微调之前，先让外部 LLM（qwen-plus / qwen-max / etc.）直接读 test 集，
  产出 explanation output，然后走一遍 explanation_eval + judge，
  作为 LoRA 微调后模型的 baseline / 上限参照。

  可选启用 Agent Loop —— 单样本级别的反思-重试：
    generate → faithfulness/judge critique → 若低于阈值把 gap 反馈进 messages 再 generate。

用法：
    python -m apps.run_external_direct_eval --config configs/experiments/external_direct_eval_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许 python apps/run_external_direct_eval.py 直接跑
_HERE = Path(__file__).resolve()
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from data_process.generate_outputs_distillation import (
    build_messages_for_entry,
    call_llm_chat,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REQUEST_RETRIES,
    DEFAULT_REQUEST_RETRY_DELAY,
    DEFAULT_REQUEST_TIMEOUT,
)
from data_process.paths import ensure_dir, report_dir
from pipelines.agent_loop import (
    AgentLoopConfig,
    AgentLoopReasoner,
    FaithfulnessCritic,
    JudgeCritic,
    save_trace_batch,
)
from pipelines.explanation_pipeline import run_explanation_eval
from pipelines.io_utils import load_config, make_run_id, save_json
from pipelines.compare_pipeline import write_compare_summary
from pipelines.query_utils import get_query_group_text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_sample_view(entry: Dict[str, Any], generated_output: str) -> Dict[str, Any]:
    """把一条 raw entry + 模型 output 归一化为 critic 需要的 sample dict。"""
    input_data = entry.get("input", {}) if isinstance(entry, dict) else {}
    return {
        "sentence": str(input_data.get("sentence", "")),
        "query_group": get_query_group_text(input_data),
        "kg_evidence": str(input_data.get("kg_evidence", "")),
        "predicted_label": str(entry.get("predicted_label", "")) or str(input_data.get("gold_label", "")),
        "gold_label": str(input_data.get("gold_label", "")),
        "output": generated_output,
    }


def _make_reasoner_fn(
    entry: Dict[str, Any],
    api_key: str,
    model_id: str,
    base_url: str,
    max_new_tokens: int,
    request_timeout: int,
    request_retries: int,
    request_retry_delay: float,
    max_kg_evidence_chars: int,
    max_sentence_chars: int,
    generation_mode: str,
    ignore_kg_evidence: bool,
):
    """为一条 entry 构造 ReasonerFn(messages, round_idx) -> str。

    第 1 轮：用 build_messages_for_entry 生成的完整提示；
    第 2+ 轮：直接吃循环拼好的 messages（包含 assistant/user 反馈 turn）。
    """
    initial_messages = build_messages_for_entry(
        entry,
        max_kg_evidence_chars=max_kg_evidence_chars,
        max_sentence_chars=max_sentence_chars,
        generation_mode=generation_mode,
        ignore_kg_evidence=ignore_kg_evidence,
    )

    def _reasoner(messages: List[Dict[str, str]], round_idx: int) -> str:
        eff_messages = initial_messages if round_idx == 1 else messages
        return call_llm_chat(
            messages=eff_messages,
            api_key=api_key,
            model=model_id,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=request_retries,
            retry_delay=request_retry_delay,
            max_output_tokens=max_new_tokens,
        )

    return _reasoner, initial_messages


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _get_section(conf: Dict[str, Any], key: str, required: bool = True) -> Dict[str, Any]:
    section = conf.get(key)
    if section is None:
        if required:
            raise ValueError(f"config missing required section: {key}")
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key} must be a dict")
    return section


def run_external_direct_eval(config_path: str) -> Dict[str, Any]:
    conf = load_config(config_path)

    exp = _get_section(conf, "experiment")
    exp_name = str(exp.get("name", "external_direct_eval"))
    output_root = str(exp.get("output_root", "data/reports"))
    run_id = make_run_id(exp_name)

    exp_dir = ensure_dir(Path(output_root) / exp_name)
    inference_dir = ensure_dir(exp_dir / "inference")

    # 快照 config 便于回溯
    save_json(str(exp_dir / "config_snapshot.json"), conf)

    # ---- 生成阶段配置 ----
    gen = _get_section(conf, "external_generation")
    input_json = str(gen["input"])
    model_id = str(gen.get("model", "qwen-plus"))
    base_url = str(gen.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    api_key = str(gen.get("api_key") or os.getenv("DASHSCOPE_API_KEY") or "")
    if not api_key:
        raise ValueError(
            "external_generation.api_key is empty and DASHSCOPE_API_KEY env var is not set"
        )
    generation_mode = str(gen.get("generation_mode", "explanation_without_kg"))
    ignore_kg = bool(gen.get("ignore_kg_evidence", False))
    max_new_tokens = int(gen.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    max_kg_chars = int(gen.get("max_kg_evidence_chars", 0))
    max_sentence_chars = int(gen.get("max_sentence_chars", 0))
    limit = gen.get("limit", None)
    limit = int(limit) if limit else None
    request_timeout = int(gen.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
    request_retries = int(gen.get("request_retries", DEFAULT_REQUEST_RETRIES))
    request_retry_delay = float(gen.get("request_retry_delay", DEFAULT_REQUEST_RETRY_DELAY))

    # ---- Agent Loop 配置 ----
    loop_conf = conf.get("agent_loop", {}) or {}
    enable_loop = bool(loop_conf.get("enabled", False))
    max_rounds = int(loop_conf.get("max_rounds", 3))
    faith_min_coverage = float(loop_conf.get("faithfulness_min_coverage", 0.4))
    faith_max_hall = float(loop_conf.get("faithfulness_max_hallucination", 0.5))
    use_judge_critic = bool(loop_conf.get("use_judge_critic", False))
    judge_min_score = float(loop_conf.get("judge_min_overall_score", 7.0))
    keep_full_trace = bool(loop_conf.get("keep_full_messages_in_trace", False))
    verbose = bool(loop_conf.get("verbose", True))

    # judge 的 api_key 复用 external_generation.api_key，允许 conf.judge 覆盖
    judge_conf = _get_section(conf, "judge", required=False)
    judge_api_key = str(judge_conf.get("api_key") or api_key or "")
    judge_model_id = str(judge_conf.get("model_id", "qwen-max"))
    judge_base_url = str(judge_conf.get("base_url", base_url))

    print(f"[external-direct-eval] experiment={exp_name}, run_id={run_id}")
    print(f"[external-direct-eval] loading input: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if limit:
        raw_data = raw_data[:limit]
    print(f"[external-direct-eval] samples: {len(raw_data)}, agent_loop={'ON' if enable_loop else 'OFF'}")

    # ---- 逐条生成 ----
    results: List[Dict[str, Any]] = []
    traces_by_id: Dict[str, Any] = {}
    generation_failures = 0
    start_t = time.time()

    for idx, entry in enumerate(raw_data):
        input_data = entry.get("input", {}) if isinstance(entry, dict) else {}
        if not get_query_group_text(input_data):
            print(f"[skip] idx={idx} has no query_group, skipping")
            continue

        reasoner_fn, initial_messages = _make_reasoner_fn(
            entry=entry,
            api_key=api_key,
            model_id=model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_retry_delay=request_retry_delay,
            max_kg_evidence_chars=max_kg_chars,
            max_sentence_chars=max_sentence_chars,
            generation_mode=generation_mode,
            ignore_kg_evidence=ignore_kg,
        )

        output_text = ""
        trace_dict: Optional[Dict[str, Any]] = None

        if enable_loop:
            # 构造 critic 列表
            critics = [
                FaithfulnessCritic(
                    min_coverage=faith_min_coverage,
                    max_hallucination=faith_max_hall,
                )
            ]
            if use_judge_critic:
                if not judge_api_key:
                    raise ValueError("agent_loop.use_judge_critic=true 但 judge.api_key 为空")
                critics.append(
                    JudgeCritic(
                        api_key=judge_api_key,
                        model_id=judge_model_id,
                        base_url=judge_base_url,
                        min_overall_score=judge_min_score,
                    )
                )

            loop = AgentLoopReasoner(
                reasoner=reasoner_fn,
                critics=critics,
                config=AgentLoopConfig(max_rounds=max_rounds, verbose=verbose),
            )
            sample_view = _build_sample_view(entry, generated_output="")
            trace = loop.run(sample=sample_view, initial_messages=initial_messages)
            output_text = trace.final_output
            from pipelines.agent_loop import trace_to_dict
            trace_dict = trace_to_dict(trace, keep_full_messages=keep_full_trace)
            traces_by_id[str(idx)] = trace
        else:
            try:
                output_text = reasoner_fn(initial_messages, 1)
            except Exception as exc:
                print(f"[error] idx={idx} generation failed: {exc}")
                output_text = ""
                generation_failures += 1

        # 记录结果，同时 pop 掉 gold_label 避免污染下游评估
        out_entry = dict(entry)
        cleaned_input = dict(input_data)
        cleaned_input.pop("gold_label", None)
        if ignore_kg:
            cleaned_input.pop("kg_evidence", None)
        out_entry["input"] = cleaned_input
        out_entry["output"] = output_text
        if trace_dict is not None:
            out_entry["agent_loop"] = {
                "n_rounds": trace_dict["n_rounds"],
                "final_passed": trace_dict["final_passed"],
                "final_score": trace_dict["final_score"],
                "stopped_reason": trace_dict["stopped_reason"],
            }
        results.append(out_entry)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(raw_data):
            elapsed = time.time() - start_t
            print(f"[external-direct-eval] progress {idx + 1}/{len(raw_data)} elapsed={elapsed:.1f}s")

    # ---- 落盘推理结果 ----
    out_path = inference_dir / "test_out.json"
    save_json(str(out_path), results)
    print(f"[external-direct-eval] saved predictions to {out_path}")

    if enable_loop and traces_by_id:
        trace_path = inference_dir / f"agent_trace_{run_id}.json"
        save_trace_batch(traces_by_id, str(trace_path), keep_full_messages=keep_full_trace)
        print(f"[external-direct-eval] saved agent-loop trace to {trace_path}")

    # ---- 直接接 explanation_eval + judge ----
    explanation_conf = conf.get("explanation_eval", {}) or {}
    if explanation_conf.get("enabled", False):
        # 允许 config 里显式给 predictions_file / labels_file / output_dir；
        # 未给则自动填成刚刚落盘的推理产物
        explanation_conf = dict(explanation_conf)
        explanation_conf.setdefault("predictions_file", str(out_path))
        explanation_conf.setdefault("output_dir", str(exp_dir / "explanation_eval"))
        judge_section = dict(judge_conf) if judge_conf else {}
        if judge_section:
            judge_section.setdefault("output_dir", str(exp_dir / "judge_eval"))
        eval_result = run_explanation_eval(explanation_conf, judge_section)
        print("[external-direct-eval] evaluation done:")
        print(json.dumps(eval_result.get("faithfulness", {}), ensure_ascii=False, indent=2))
        if "judge" in eval_result:
            print(json.dumps(eval_result["judge"], ensure_ascii=False, indent=2))
    else:
        print("[external-direct-eval] explanation_eval.enabled=false, skip evaluation")
        eval_result = {}

    # Keep the direct baseline compatible with the shared result aggregator and
    # make it directly comparable with LoRA/distilled experiments.
    compare_rows: List[Dict[str, Any]] = []
    if eval_result:
        faith = eval_result.get("faithfulness", {}) or {}
        judge = eval_result.get("judge", {}) or {}
        compare_rows.append(
            {
                "experiment_name": exp_name,
                "model_name": str(explanation_conf.get("model_name", model_id)),
                "track": "explanation",
                "coverage_mean": faith.get("coverage_mean", ""),
                "hallucination_mean": faith.get("hallucination_mean", ""),
                "consistency_mean": faith.get("consistency_mean", ""),
                "mechanism_overall_score_mean": judge.get("mechanism_overall_score_mean", ""),
            }
        )

    compare_csv = ""
    if compare_rows:
        compare_csv = write_compare_summary(str(exp_dir / "compare_summary.csv"), compare_rows)
        print(f"[external-direct-eval] saved comparison summary to {compare_csv}")

    run_summary = {
        "run_id": run_id,
        "experiment": exp_name,
        "n_samples": len(results),
        "generation": {
            "requested_samples": len(raw_data),
            "successful_outputs": sum(1 for item in results if str(item.get("output", "")).strip()),
            "failed_samples": generation_failures,
        },
        "agent_loop": {
            "enabled": enable_loop,
            "max_rounds": max_rounds,
            "use_judge_critic": use_judge_critic,
        },
        "external_model": {"model": model_id, "base_url": base_url},
        "artifacts": {
            "predictions": str(out_path),
            "compare_summary": compare_csv,
        },
        "evaluation_summary": eval_result,
    }
    save_json(str(exp_dir / "run_summary.json"), run_summary)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External-LLM direct evaluation (skips fine-tuning); optional agent-loop critique"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_external_direct_eval(args.config)


if __name__ == "__main__":
    main()
