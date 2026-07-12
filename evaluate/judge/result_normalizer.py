import json
import re
from typing import Any, Dict, List


RUBRIC_FIELDS = (
    "mechanism_chain_completeness",
    "mechanism_direction_correctness",
    "mechanism_granularity",
    "mechanism_internal_consistency",
    "uncertainty_calibration",
    "clinical_actionability",
)


def _clip_int(value: Any, lower: int, upper: int, default: int = 0) -> int:
    try:
        number = int(value)
    except Exception:
        number = default
    return max(lower, min(upper, number))


def _extract_json_payload(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _normalize_claim_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _decision_from_score(score: int) -> str:
    if score >= 10:
        return "good"
    if score >= 7:
        return "fair"
    return "poor"


def normalize_judge_result(payload: Any, raw_text: str = "") -> Dict[str, Any]:
    if isinstance(payload, dict):
        parsed = payload
    else:
        parsed = _extract_json_payload(str(payload or raw_text or ""))

    rubric: Dict[str, int] = {}
    for key in RUBRIC_FIELDS:
        rubric[key] = _clip_int(parsed.get(key, 0), 0, 2, default=0)

    overall_score = parsed.get("overall_score", parsed.get("mechanism_overall_score"))
    if overall_score is None:
        overall_score = sum(rubric.values())
    overall_score = _clip_int(overall_score, 0, 12, default=sum(rubric.values()))

    mechanism_gaps = _normalize_claim_list(parsed.get("mechanism_gaps", parsed.get("gaps", [])))
    short_rationale = str(parsed.get("short_rationale", "") or "").strip()
    if not short_rationale and raw_text:
        short_rationale = raw_text[:200]

    return {
        "mechanism_chain_completeness": rubric["mechanism_chain_completeness"],
        "mechanism_direction_correctness": rubric["mechanism_direction_correctness"],
        "mechanism_granularity": rubric["mechanism_granularity"],
        "mechanism_internal_consistency": rubric["mechanism_internal_consistency"],
        "uncertainty_calibration": rubric["uncertainty_calibration"],
        "clinical_actionability": rubric["clinical_actionability"],
        "mechanism_overall_score": overall_score,
        "mechanism_overall_decision": str(
            parsed.get("mechanism_overall_decision", parsed.get("overall_decision", ""))
            or _decision_from_score(overall_score)
        ),
        "mechanism_gaps": mechanism_gaps,
        "judge_short_rationale": short_rationale,
        "rationale": short_rationale,
    }
