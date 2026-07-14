import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pipelines.query_utils import expand_query_pairs


LABEL_ALIASES = {
    "mechanism": "mechanism",
    "effect": "effect",
    "advice": "advise",
    "advise": "advise",
    "int": "int",
    "interaction": "int",
    "false": "false",
    "none": "false",
}

LABEL_BACKFILL_PATTERNS: List[Tuple[str, List[str]]] = [
    (
        "false",
        [
            r"\bno interaction\b",
            r"\bnot interact",
            r"\bnot known\b",
            r"\bunknown\b",
            r"\buncertain\b",
            r"\bno empirical evidence\b",
        ],
    ),
    (
        "advise",
        [
            r"\bavoid\b",
            r"\bmonitor\b",
            r"\bdose adjustment\b",
            r"\bcontraindicat",
            r"\bwarning\b",
            r"\brecommend\b",
        ],
    ),
    (
        "mechanism",
        [
            r"\breduce(?:s|d|ing)? metabolism\b",
            r"\binhibit(?:s|ed|ing)? metabolism\b",
            r"\bmetabolism\b",
            r"\bclearance\b",
            r"\benzyme\b",
            r"\bcyp\b",
            r"\binduction\b",
            r"\binhibition\b",
        ],
    ),
    (
        "effect",
        [
            r"\bincrease(?:s|d|ing)? (?:plasma )?(?:concentration|exposure|level|toxicity|response)\b",
            r"\bdecrease(?:s|d|ing)? (?:plasma )?(?:concentration|exposure|level|response)\b",
            r"\bpharmacodynamic\b",
            r"\boutcome\b",
        ],
    ),
    ("int", [r"\binteract(?:ion|s|ed|ing)?\b", r"\bdrug[- ]drug interaction\b"]),
]

OPTIONAL_INFERENCE_HINTS = {
    "may",
    "might",
    "could",
    "possible",
    "potential",
    "plausible",
    "likely",
    "suggests",
    "reflects",
    "implies",
    "risk",
    "adverse",
    "hypoglycemia",
    "cyp",
    "cytochrome",
    "hepatic",
    "enzyme",
    "transporter",
    "substrate",
    "pathway",
}

PAIR_CONNECTORS = ("->", "=>")


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label_text(value: Any) -> str:
    key = str(value or "").strip().lower()
    return LABEL_ALIASES.get(key, "")


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "may", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "this", "those", "to", "was", "were", "with",
    "sentence", "sentences", "text", "query", "queries", "evidence", "clinical", "phenomenon",
    "interpretation", "mechanistic", "mechanism", "summary", "analysis", "steps", "step",
    "explanation", "global_phenomenon", "representative_pairs", "confidence", "assessment",
    "describe", "describes", "described", "reflect", "reflects", "indicate", "indicates",
    "observed", "observe", "identified", "identify", "specifies", "specify", "defined",
    "requires", "require", "especially", "during", "overall", "therefore", "thus",
    "coadministration", "co", "administration", "concurrent", "use", "used", "using",
    "likely", "suggest", "suggests", "imply", "implies", "not", "vice", "versa",
}


def _normalize_token(token: str) -> str:
    tok = str(token or "").strip().lower()
    tok = tok.strip("_'")
    if not tok:
        return ""

    irregular = {
        "reduces": "reduce",
        "reduced": "reduce",
        "reducing": "reduce",
        "reduction": "reduce",
        "increases": "increase",
        "increased": "increase",
        "increasing": "increase",
        "decreases": "decrease",
        "decreased": "decrease",
        "decreasing": "decrease",
        "concentrations": "concentration",
        "levels": "level",
        "effects": "effect",
        "metabolic": "metabolism",
        "metabolized": "metabolism",
        "metabolised": "metabolism",
        "metabolizing": "metabolism",
        "inhibition": "inhibit",
        "inhibits": "inhibit",
        "inhibited": "inhibit",
        "inhibitor": "inhibit",
        "induction": "induce",
        "induces": "induce",
        "induced": "induce",
        "monitoring": "monitor",
        "cautious": "caution",
        "cautioned": "caution",
    }
    if tok in irregular:
        return irregular[tok]

    if len(tok) > 5 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 5 and tok.endswith("ing"):
        tok = tok[:-3]
    elif len(tok) > 4 and tok.endswith("ed"):
        tok = tok[:-2]
    elif len(tok) > 4 and tok.endswith("es"):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s"):
        tok = tok[:-1]
    return tok


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    tokens = [_normalize_token(tok) for tok in re.findall(r"[a-z0-9_]+", normalize_text(text))]
    tokens = [tok for tok in tokens if tok]
    if not drop_stopwords:
        return tokens
    return [tok for tok in tokens if tok not in STOPWORDS and not tok.isdigit()]


def strip_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _clean_claim_text(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace("\r", "\n")
    s = re.sub(r"(?<![A-Za-z])\d+\s*[\.\)]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r-,:;[]{}\"'")
    return s.strip()


def _is_noise_claim(text: str) -> bool:
    s = _clean_claim_text(text)
    if not s:
        return True
    if re.fullmatch(r"[\[\]\{\}\",:;]+", s):
        return True
    tokens = tokenize(s)
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0] in {"json", "list", "object", "analysis_step", "representative_pair"}:
        return True
    if s in {"analysis_steps", "representative_pairs", "mechanism_summary", "query"}:
        return True
    return False


def split_claims(text: str) -> List[str]:
    cleaned = strip_code_fence(text)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"(?<=\S)\s+(?=\d+\s*[\.\)])", "\n", cleaned)
    raw_parts = re.split(r"[\n;]+", cleaned)
    claims: List[str] = []
    for part in raw_parts:
        piece = _clean_claim_text(part)
        if _is_noise_claim(piece):
            continue
        claims.append(piece)
    return claims


def infer_label_from_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    for label, patterns in LABEL_BACKFILL_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return label
    return ""


def _parse_json_payload(text: str) -> Optional[Any]:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"(\[\s*\{.*\}\s*\]|\{\s*\".*\}\s*)", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _split_sentence_units(text: str) -> List[str]:
    cleaned = strip_code_fence(text)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"(?<![A-Za-z])\d+\s*[\.\)]\s*", "\n", cleaned)
    raw_parts = re.split(r"[\n;]+|(?<=[.!?])\s+", cleaned)
    parts: List[str] = []
    for part in raw_parts:
        piece = _clean_claim_text(part)
        if _is_noise_claim(piece):
            continue
        parts.append(piece)
    return parts


def _text_to_claim_object(text: str, claim_type: str = "core_claim") -> Dict[str, str]:
    return {"type": claim_type, "text": _clean_claim_text(text)}


def _classify_claim_bucket(text: str) -> str:
    tokens = set(tokenize(text, drop_stopwords=False))
    if tokens & OPTIONAL_INFERENCE_HINTS:
        return "optional_inference"
    return "core_claim"


def _dedupe_claim_objects(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in items:
        text = _clean_claim_text(item.get("text", ""))
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"type": str(item.get("type", "")), "text": text})
    return deduped


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not value:
        return []
    return [str(value).strip()]


def _normalize_pairs(pairs: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in pairs:
        text = str(item or "").strip()
        if not text:
            continue
        canon = normalize_text(text)
        if canon in seen:
            continue
        seen.add(canon)
        ordered.append(text)
    return ordered


def _normalize_legacy_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    core_claims: List[Dict[str, str]] = []
    optional_inference: List[Dict[str, str]] = []
    analysis_steps: List[str] = []
    mechanism_summaries: List[str] = []
    representative_pairs: List[str] = []
    queries: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "") or "").strip()
        if query:
            queries.append(query)
        representative_pairs.extend(_string_list(item.get("representative_pairs")))

        for sentence in _split_sentence_units(str(item.get("analysis_steps", "") or "")):
            analysis_steps.append(sentence)
            bucket = _classify_claim_bucket(sentence)
            target = optional_inference if bucket == "optional_inference" else core_claims
            target.append(_text_to_claim_object(sentence, bucket))

        mechanism_summary = str(item.get("mechanism_summary", "") or "").strip()
        if mechanism_summary:
            mechanism_summaries.append(mechanism_summary)
            bucket = _classify_claim_bucket(mechanism_summary)
            target = optional_inference if bucket == "optional_inference" else core_claims
            target.append(_text_to_claim_object(mechanism_summary, bucket))

    merged_summary = " ".join(mechanism_summaries).strip()
    merged_steps = " ".join(analysis_steps).strip()

    return {
        "predicted_label": infer_label_from_text(f"{merged_summary} {merged_steps}"),
        "explanation_type": "legacy_list",
        "core_claims": _dedupe_claim_objects(core_claims),
        "optional_inference": _dedupe_claim_objects(optional_inference),
        "mechanism_summary": merged_summary,
        "analysis_steps": analysis_steps,
        "representative_pairs": _normalize_pairs(representative_pairs),
        "query_items": _normalize_pairs(queries),
        "evidence_sent_ids": [],
        "parse_mode": "json_list",
    }


def _normalize_structured_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    core_claims: List[Dict[str, str]] = []
    optional_inference: List[Dict[str, str]] = []

    core_items = payload.get("core_claims", [])
    if isinstance(core_items, list):
        for item in core_items:
            if isinstance(item, dict):
                core_claims.append(
                    _text_to_claim_object(item.get("text", ""), str(item.get("type", "core_claim") or "core_claim"))
                )
            else:
                core_claims.append(_text_to_claim_object(item, "core_claim"))

    optional_items = payload.get("optional_inference", [])
    if isinstance(optional_items, list):
        for item in optional_items:
            if isinstance(item, dict):
                optional_inference.append(
                    _text_to_claim_object(
                        item.get("text", ""), str(item.get("type", "optional_inference") or "optional_inference")
                    )
                )
            else:
                optional_inference.append(_text_to_claim_object(item, "optional_inference"))

    analysis_steps = _split_sentence_units(str(payload.get("analysis_steps", "") or ""))
    mechanism_summary = str(payload.get("mechanism_summary", "") or "").strip()

    if not core_claims:
        for sentence in analysis_steps:
            bucket = _classify_claim_bucket(sentence)
            target = optional_inference if bucket == "optional_inference" else core_claims
            target.append(_text_to_claim_object(sentence, bucket))

    if mechanism_summary and not core_claims:
        bucket = _classify_claim_bucket(mechanism_summary)
        target = optional_inference if bucket == "optional_inference" else core_claims
        target.append(_text_to_claim_object(mechanism_summary, bucket))

    evidence_ids = payload.get("evidence_sent_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []

    return {
        "predicted_label": normalize_label_text(payload.get("predicted_label", "")),
        "explanation_type": str(payload.get("explanation_type", "structured_dict") or "structured_dict"),
        "core_claims": _dedupe_claim_objects(core_claims),
        "optional_inference": _dedupe_claim_objects(optional_inference),
        "mechanism_summary": mechanism_summary,
        "analysis_steps": analysis_steps,
        "representative_pairs": _normalize_pairs(_string_list(payload.get("representative_pairs"))),
        "query_items": _normalize_pairs(_string_list(payload.get("query_items"))),
        "evidence_sent_ids": [int(x) for x in evidence_ids if str(x).isdigit()],
        "parse_mode": "json_dict",
    }


def normalize_explanation_output(text: str, predicted_label: str = "") -> Dict[str, Any]:
    parsed = _parse_json_payload(text)
    if isinstance(parsed, list):
        dict_items = [item for item in parsed if isinstance(item, dict)]
        normalized = _normalize_legacy_items(dict_items)
    elif isinstance(parsed, dict):
        normalized = _normalize_structured_dict(parsed)
    else:
        plain_claims = [_text_to_claim_object(item, "plain_text") for item in _split_sentence_units(text)]
        normalized = {
            "predicted_label": "",
            "explanation_type": "plain_text",
            "core_claims": _dedupe_claim_objects(plain_claims),
            "optional_inference": [],
            "mechanism_summary": "",
            "analysis_steps": [item["text"] for item in plain_claims],
            "representative_pairs": [],
            "query_items": [],
            "evidence_sent_ids": [],
            "parse_mode": "raw_text_fallback",
        }

    original_label = normalize_label_text(predicted_label)
    inferred_label = normalize_label_text(normalized.get("predicted_label", "")) or infer_label_from_text(
        " ".join(
            [str(text or ""), normalized.get("mechanism_summary", "")]
            + [item.get("text", "") for item in normalized.get("core_claims", [])]
            + [item.get("text", "") for item in normalized.get("optional_inference", [])]
        )
    )
    resolved_label = original_label or inferred_label

    normalized["predicted_label"] = resolved_label
    normalized["label_backfilled"] = bool(resolved_label and not original_label)
    normalized["label_source"] = "input" if original_label else ("explanation_backfill" if resolved_label else "")
    return normalized


def extract_claim_records(text: str) -> List[Dict[str, str]]:
    normalized = normalize_explanation_output(text)
    records: List[Dict[str, str]] = []
    seen = set()

    for item in normalized.get("core_claims", []):
        claim = _clean_claim_text(item.get("text", ""))
        key = ("core_claims", normalize_text(claim))
        if claim and key not in seen:
            seen.add(key)
            records.append({"query": "", "field": "core_claims", "claim": claim, "claim_type": "core_claim"})

    mechanism_summary = _clean_claim_text(normalized.get("mechanism_summary", ""))
    if mechanism_summary and not records:
        key = ("mechanism_summary", normalize_text(mechanism_summary))
        if key not in seen:
            seen.add(key)
            records.append(
                {
                    "query": "",
                    "field": "mechanism_summary",
                    "claim": mechanism_summary,
                    "claim_type": "core_claim",
                }
            )

    if not records:
        for claim in normalized.get("analysis_steps", []):
            cleaned = _clean_claim_text(claim)
            key = ("analysis_steps", normalize_text(cleaned))
            if cleaned and key not in seen:
                seen.add(key)
                records.append(
                    {
                        "query": "",
                        "field": "analysis_steps",
                        "claim": cleaned,
                        "claim_type": "analysis_steps",
                    }
                )

    if not records:
        for claim in split_claims(text):
            records.append({"query": "", "field": "plain_text", "claim": claim, "claim_type": "plain_text"})
    return records


def parse_query_pairs(queries: str) -> List[str]:
    return _normalize_pairs(expand_query_pairs(queries))


def compute_query_coverage(queries: str, representative_pairs: List[str]) -> Dict[str, Any]:
    query_pairs = parse_query_pairs(queries)
    rep_pairs = _normalize_pairs([str(x or "").replace("=>", "->") for x in representative_pairs])
    query_set = {normalize_text(item) for item in query_pairs}
    rep_set = {normalize_text(item) for item in rep_pairs}
    covered = sorted(query_set & rep_set)
    return {
        "query_pairs": query_pairs,
        "representative_pairs": rep_pairs,
        "covered_queries": len(covered),
        "total_queries": len(query_pairs),
        "query_coverage": float(len(covered) / len(query_pairs)) if query_pairs else 0.0,
    }


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
