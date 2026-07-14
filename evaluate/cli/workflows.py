from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluate.core.data_loader import load_predictions_json, load_test_data
from evaluate.explanation.faithfulness import aggregate_faithfulness, compute_faithfulness_for_sample
from evaluate.judge.qwen_judge import QwenMaxJudge
from evaluate.results_writer import now_tag, write_json, write_predictions_tsv
from pipelines.query_utils import get_query_group_text


def _normalize_label_text(value: Any) -> str:
    mapping = {
        "mechanism": "mechanism",
        "effect": "effect",
        "advice": "advise",
        "advise": "advise",
        "int": "int",
        "false": "false",
    }
    key = str(value or "").strip().lower()
    return mapping.get(key, "")


def _extract_label_from_output_field(output_text: Any) -> str:
    text = str(output_text or "").strip()
    if not text:
        return ""

    # 1) Try strict JSON list/object parsing first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            labels: List[str] = []
            for item in parsed:
                if isinstance(item, dict):
                    lb = _normalize_label_text(item.get("label"))
                    if lb:
                        labels.append(lb)
            if labels:
                counts: Dict[str, int] = {}
                for lb in labels:
                    counts[lb] = counts.get(lb, 0) + 1
                return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        if isinstance(parsed, dict):
            return _normalize_label_text(parsed.get("label"))
    except Exception:
        pass

    # 2) Regex fallback for JSON-like outputs.
    m = re.search(r'"label"\s*:\s*"(Mechanism|Effect|Advice|Advise|Int|False)"', text, flags=re.IGNORECASE)
    if m:
        return _normalize_label_text(m.group(1))

    # 3) Plain text label fallback.
    return _normalize_label_text(text)


def _rows_from_classification(samples, pred_result) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    predictions = pred_result["predictions"]
    reasoning = pred_result["reasoning_chains"]
    for idx, (true_label, text) in enumerate(samples):
        rows.append(
            {
                "id": idx,
                "true_label": true_label,
                "predicted_label": predictions[idx],
                "text": text,
                "reasoning": reasoning[idx],
                "sentence": text,
                "query_group": "",
                "queries": "",
                "kg_evidence": "",
            }
        )
    return rows


def run_classify(config) -> Dict[str, Any]:
    from evaluate.classification.ddi_classifier import DDIClassifier
    from evaluate.core.metrics import compute_classification_metrics

    samples = load_test_data(config.test_data_path)
    classifier = DDIClassifier(
        model_id=config.model_id,
        ddi_types=config.ddi_types,
        max_new_tokens=config.max_new_tokens,
    )
    pred_result = classifier.predict_dataset(samples)

    rows = _rows_from_classification(samples, pred_result)
    y_true = [x[0] for x in samples]
    y_pred = pred_result["predictions"]
    metrics = compute_classification_metrics(y_true, y_pred, config.ddi_types)

    tag = now_tag()
    out_dir = Path(config.output_dir)
    pred_path = out_dir / f"test_predict_{tag}.csv"
    metrics_path = out_dir / f"classification_metrics_{tag}.json"
    write_predictions_tsv(pred_path, rows)
    write_json(metrics_path, metrics)

    return {
        "rows": rows,
        "metrics": metrics,
        "tag": tag,
        "artifacts": [str(pred_path), str(metrics_path)],
    }


def _row_query_group_text(row: Dict[str, Any]) -> str:
    value = str(row.get("query_group", "") or "").strip()
    if value:
        return value
    return str(row.get("queries", "") or "")


def _rows_from_predictions_file(pred_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(pred_items):
        input_data = item.get("input", {}) if isinstance(item, dict) else {}
        predicted_label = _normalize_label_text(item.get("predicted_label", ""))
        if not predicted_label:
            predicted_label = _normalize_label_text(input_data.get("predicted_label", ""))
        if not predicted_label:
            predicted_label = _normalize_label_text(input_data.get("target_label", ""))
        if not predicted_label:
            predicted_label = _extract_label_from_output_field(item.get("output", ""))

        true_label = _normalize_label_text(item.get("gold_label", ""))
        if not true_label:
            true_label = _normalize_label_text(input_data.get("gold_label", ""))
        if not true_label:
            true_label = _normalize_label_text(input_data.get("target_label", ""))

        rows.append(
            {
                "id": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_label_source": "prediction" if predicted_label else "",
                "text": str(input_data.get("sentence", "")),
                "reasoning": str(item.get("output", "")),
                "sentence": str(input_data.get("sentence", "")),
                "query_group": get_query_group_text(input_data),
                "queries": str(input_data.get("queries", "") or get_query_group_text(input_data)),
                "kg_evidence": str(input_data.get("kg_evidence", "")),
            }
        )
    return rows


def _make_sentence_query_key(sentence: str, query_group: str) -> Tuple[str, str]:
    return (str(sentence or "").strip(), str(query_group or "").strip())


def _build_predicted_label_lookup(pred_items: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for item in pred_items:
        if not isinstance(item, dict):
            continue
        input_data = item.get("input", {}) if isinstance(item.get("input", {}), dict) else {}
        label = _normalize_label_text(item.get("predicted_label", ""))
        if not label:
            label = _normalize_label_text(input_data.get("predicted_label", ""))
        if not label:
            label = _extract_label_from_output_field(item.get("output", ""))
        if not label:
            continue
        key = _make_sentence_query_key(input_data.get("sentence", ""), get_query_group_text(input_data))
        if key[0] or key[1]:
            lookup[key] = label
    return lookup


def _candidate_label_prediction_paths(predictions_file: str) -> List[str]:
    pred_path = Path(predictions_file)
    name = pred_path.name
    candidates: List[Path] = []
    replacements = [
        ("reasoning_without_kg", "label_only_without_kg"),
        ("explanation_without_kg", "label_only_without_kg"),
        ("reasoning_with_kg", "label_only_with_kg"),
        ("explanation_with_kg", "label_only_with_kg"),
    ]
    for src, dst in replacements:
        if src in name:
            candidates.append(pred_path.with_name(name.replace(src, dst)))

    # legacy fallback：兼容老实验产出还留在 data/reports/_legacy_test_predictions/ 的情况
    fallback_roots = [
        Path("data/reports/_legacy_test_predictions"),
        Path("results/test_predictions"),  # legacy path，兼容旧机器
        pred_path.parent,
        pred_path.parent.parent,
    ]
    for src, dst in replacements:
        if src in name:
            target_name = name.replace(src, dst)
            for root in fallback_roots:
                candidates.append(root / target_name)

    seen = set()
    ordered: List[str] = []
    for cand in candidates:
        resolved = str(cand)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _resolve_label_predictions_file(config) -> Optional[str]:
    explicit = getattr(config, "label_predictions_file", None)
    if explicit:
        return str(explicit)
    predictions_file = getattr(config, "predictions_file", None)
    if not predictions_file:
        return None
    for candidate in _candidate_label_prediction_paths(str(predictions_file)):
        if Path(candidate).exists():
            return candidate
    return None


def _apply_predicted_label_backfill(rows: List[Dict[str, Any]], pred_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows or not pred_items:
        return rows

    label_lookup = _build_predicted_label_lookup(pred_items)
    for row in rows:
        if row.get("predicted_label"):
            continue
        key = _make_sentence_query_key(row.get("sentence", ""), _row_query_group_text(row))
        label = label_lookup.get(key, "")
        if label:
            row["predicted_label"] = label
            row["predicted_label_source"] = "label_prediction_file"

    n = min(len(rows), len(pred_items))
    for idx in range(n):
        if rows[idx].get("predicted_label"):
            continue
        item = pred_items[idx]
        input_data = item.get("input", {}) if isinstance(item.get("input", {}), dict) else {}
        label = _normalize_label_text(item.get("predicted_label", ""))
        if not label:
            label = _normalize_label_text(input_data.get("predicted_label", ""))
        if not label:
            label = _extract_label_from_output_field(item.get("output", ""))
        if label:
            rows[idx]["predicted_label"] = label
            rows[idx]["predicted_label_source"] = "label_prediction_file_index_fallback"
    return rows


def _apply_labels_sidecar(rows: List[Dict[str, Any]], labels_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows or not labels_items:
        return rows

    # Primary: key-based match by sentence+queries to avoid index drift.
    label_by_key: Dict[Any, str] = {}
    for item in labels_items:
        lb = _normalize_label_text(item.get("gold_label", ""))
        if not lb:
            continue
        key = (str(item.get("sentence", "")), get_query_group_text(item))
        if key[0] or key[1]:
            label_by_key[key] = lb

    for row in rows:
        if row.get("true_label"):
            continue
        key = (str(row.get("sentence", "")), _row_query_group_text(row))
        lb = label_by_key.get(key, "")
        if lb:
            row["true_label"] = lb

    # Fallback: index-based fill for remaining blanks.
    n = min(len(rows), len(labels_items))
    for i in range(n):
        if rows[i].get("true_label"):
            continue
        lb = _normalize_label_text(labels_items[i].get("gold_label", ""))
        if lb:
            rows[i]["true_label"] = lb
    return rows


def _resolve_rows_for_analysis(config, current_rows: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if current_rows:
        return current_rows
    if not config.predictions_file:
        raise ValueError("predictions_file is required for faithfulness/judge-only mode")
    pred_items = load_predictions_json(config.predictions_file)
    rows = _rows_from_predictions_file(pred_items)
    labels_path = getattr(config, "predictions_labels_file", None)
    if labels_path:
        try:
            labels_items = load_predictions_json(labels_path)
            rows = _apply_labels_sidecar(rows, labels_items)
        except Exception as exc:
            print(f"[Warn] failed to load labels sidecar '{labels_path}': {exc}")
    label_predictions_path = _resolve_label_predictions_file(config)
    if label_predictions_path:
        try:
            label_pred_items = load_predictions_json(label_predictions_path)
            rows = _apply_predicted_label_backfill(rows, label_pred_items)
        except Exception as exc:
            print(f"[Warn] failed to load label predictions '{label_predictions_path}': {exc}")
    return rows


def run_faithfulness(config, rows: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = _resolve_rows_for_analysis(config, rows)

    detail: List[Dict[str, Any]] = []
    for row in rows:
        metric = compute_faithfulness_for_sample(
            sentence=row.get("sentence", ""),
            explanation=row.get("reasoning", ""),
            predicted_label=row.get("predicted_label", ""),
            kg_evidence=row.get("kg_evidence", ""),
            queries=_row_query_group_text(row),
        )
        row.update(metric)
        detail.append(metric)

    summary = aggregate_faithfulness(detail)

    tag = now_tag()
    out_dir = Path(config.output_dir)
    detail_path = out_dir / f"faithfulness_detail_{tag}.json"
    summary_path = out_dir / f"faithfulness_summary_{tag}.json"
    write_json(detail_path, rows)
    write_json(summary_path, summary)

    return {
        "rows": rows,
        "summary": summary,
        "tag": tag,
        "artifacts": [str(detail_path), str(summary_path)],
    }


def run_judge(config, rows: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = _resolve_rows_for_analysis(config, rows)
    judge = QwenMaxJudge(
        api_key=config.judge_api_key,
        model_id=config.judge_model_id,
        base_url=config.judge_base_url,
        max_retries=config.judge_max_retries,
        retry_delay=config.judge_retry_delay,
        verbose=config.judge_verbose,
    )
    judge_results = judge.batch_judge(rows, checkpoint_every=config.judge_checkpoint_every)

    for row, result in zip(rows, judge_results):
        row.update(result)

    mechanism_overall_score_mean = 0.0
    mechanism_chain_completeness_mean = 0.0
    mechanism_direction_correctness_mean = 0.0
    mechanism_granularity_mean = 0.0
    mechanism_internal_consistency_mean = 0.0
    uncertainty_calibration_mean = 0.0
    clinical_actionability_mean = 0.0
    good_rate = 0.0
    if judge_results:
        n = len(judge_results)
        mechanism_overall_score_mean = sum(float(r.get("mechanism_overall_score", 0.0)) for r in judge_results) / n
        mechanism_chain_completeness_mean = sum(float(r.get("mechanism_chain_completeness", 0.0)) for r in judge_results) / n
        mechanism_direction_correctness_mean = sum(float(r.get("mechanism_direction_correctness", 0.0)) for r in judge_results) / n
        mechanism_granularity_mean = sum(float(r.get("mechanism_granularity", 0.0)) for r in judge_results) / n
        mechanism_internal_consistency_mean = sum(float(r.get("mechanism_internal_consistency", 0.0)) for r in judge_results) / n
        uncertainty_calibration_mean = sum(float(r.get("uncertainty_calibration", 0.0)) for r in judge_results) / n
        clinical_actionability_mean = sum(float(r.get("clinical_actionability", 0.0)) for r in judge_results) / n
        good_rate = sum(1 for r in judge_results if str(r.get("mechanism_overall_decision", "")).lower() == "good") / n
    summary = {
        "n_samples": len(judge_results),
        "judge_model": config.judge_model_id,
        "mechanism_overall_score_mean": float(mechanism_overall_score_mean),
        "mechanism_chain_completeness_mean": float(mechanism_chain_completeness_mean),
        "mechanism_direction_correctness_mean": float(mechanism_direction_correctness_mean),
        "mechanism_granularity_mean": float(mechanism_granularity_mean),
        "mechanism_internal_consistency_mean": float(mechanism_internal_consistency_mean),
        "uncertainty_calibration_mean": float(uncertainty_calibration_mean),
        "clinical_actionability_mean": float(clinical_actionability_mean),
        "mechanism_good_rate": float(good_rate),
    }

    tag = now_tag()
    out_dir = Path(config.output_dir)
    detail_path = out_dir / f"judge_detail_{tag}.json"
    summary_path = out_dir / f"judge_summary_{tag}.json"
    write_json(detail_path, rows)
    write_json(summary_path, summary)

    return {
        "rows": rows,
        "summary": summary,
        "tag": tag,
        "artifacts": [str(detail_path), str(summary_path)],
    }


def run_full(config) -> Dict[str, Any]:
    classify_result = run_classify(config)
    base_rows = classify_result["rows"]

    faith_result = run_faithfulness(config, rows=[dict(row) for row in base_rows])
    faith_rows = faith_result["rows"]

    if config.use_judge:
        judge_result = run_judge(config, rows=[dict(row) for row in base_rows])
        judge_rows = judge_result["rows"]
        judge_summary = judge_result["summary"]
    else:
        judge_rows = [dict(row) for row in base_rows]
        judge_summary = {"skipped": True}

    rows: List[Dict[str, Any]] = []
    for idx, base_row in enumerate(base_rows):
        merged_row = dict(base_row)
        if idx < len(faith_rows):
            merged_row.update({k: v for k, v in faith_rows[idx].items() if k not in merged_row or k not in {"id", "true_label", "predicted_label", "text", "reasoning", "sentence", "query_group", "queries", "kg_evidence"}})
        if idx < len(judge_rows):
            merged_row.update(judge_rows[idx])
        rows.append(merged_row)

    tag = now_tag()
    out_dir = Path(config.output_dir)
    full_report = out_dir / f"full_report_{tag}.csv"
    full_summary = out_dir / f"full_summary_{tag}.json"
    write_predictions_tsv(full_report, rows)
    write_json(
        full_summary,
        {
            "classification": classify_result["metrics"],
            "faithfulness": faith_result["summary"],
            "judge": judge_summary,
        },
    )

    return {
        "classification": classify_result,
        "faithfulness": faith_result,
        "judge": judge_summary,
        "tag": tag,
        "artifacts": [str(full_report), str(full_summary)],
    }
