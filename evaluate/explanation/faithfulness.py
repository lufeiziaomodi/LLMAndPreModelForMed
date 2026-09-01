import re
from typing import Any, Dict, List, Set

from .processor import (
    compute_query_coverage,
    extract_claim_records,
    jaccard,
    normalize_explanation_output,
    normalize_label_text,
    normalize_text,
    tokenize,
)

LABEL_KEYWORDS = {
    "mechanism": {"cyp", "metabolism", "inhibition", "induction", "absorption", "transporter"},
    "effect": {"effect", "toxicity", "concentration", "exposure", "response", "outcome"},
    "advise": {"avoid", "contraindicated", "warning", "recommend", "monitor"},
    "int": {"interact", "interaction"},
    "false": {"no interaction", "not interact", "none"},
}

DISCOURSE_TOKENS = {
    "about", "all", "broad", "broadly", "candidate", "compress", "compression", "core",
    "define", "directionality", "follow", "followed", "form", "frame", "framing", "ground",
    "grounded", "link", "linking", "pair", "pairwise", "same", "scope", "signal", "state",
    "stated", "timing", "variant", "variants", "whether",
}

INFERENCE_TOKENS = {
    "absorption", "adverse", "affect", "bioavailability", "caution", "clearance", "concentration",
    "contraindicate", "drug", "effect", "elimination", "enzyme", "exposure", "half", "inhibit",
    "induce", "interaction", "level", "metabolism", "monitor", "outcome", "pair", "pharmacodynamic",
    "pharmacokinetic", "plasma", "potentiate", "recommend", "response", "risk", "serum", "substrate",
    "synergism", "synergistic", "toxicity", "transfer", "transport", "transporter", "warning",
}

MECHANISM_HINTS = {
    "cyp", "cytochrome", "enzyme", "metabolism", "transporter", "substrate",
    "inhibit", "induce", "clearance", "hepatic",
}

CLINICAL_OUTCOME_HINTS = {
    "hypoglycemia", "toxicity", "adverse", "risk", "warning", "serum",
    "outcome", "response", "event",
}

KG_ANCHOR_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]{1,40})\s*\("
    r"(?:Enzyme|Transporter|Target|Carrier|Gene/protein(?:;[^)]*)?)\)",
    flags=re.IGNORECASE,
)


def _extract_kg_anchors(kg_evidence: str) -> Set[str]:
    anchors: Set[str] = set()
    for raw_anchor in KG_ANCHOR_PATTERN.findall(str(kg_evidence or "")):
        anchors.update(tokenize(raw_anchor, drop_stopwords=False))
    return anchors


def _claim_kg_anchor_overlap(claim: str, kg_anchors: Set[str]) -> List[str]:
    if not kg_anchors:
        return []
    claim_tokens = set(tokenize(claim, drop_stopwords=False))
    return sorted(claim_tokens & kg_anchors)


def _claim_support_score(claim: str, evidence_text: str) -> float:
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence_text)
    return jaccard(claim_tokens, evidence_tokens)


def _claim_support_features(claim: str, evidence_text: str) -> Dict[str, Any]:
    normalized_claim = normalize_text(claim)
    normalized_evidence = normalize_text(evidence_text)
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence_text)
    evidence_set = set(evidence_tokens)
    overlap = [tok for tok in claim_tokens if tok in evidence_set]
    overlap_count = len(overlap)
    precision = float(overlap_count / len(claim_tokens)) if claim_tokens else 0.0
    jacc = jaccard(claim_tokens, evidence_tokens)
    exact_substring = bool(normalized_claim and normalized_claim in normalized_evidence)
    unsupported_tokens = [tok for tok in claim_tokens if tok not in evidence_set]
    discourse_unsupported = [tok for tok in unsupported_tokens if tok in DISCOURSE_TOKENS]
    inference_unsupported = [tok for tok in unsupported_tokens if tok in INFERENCE_TOKENS]
    novel_unsupported = [
        tok for tok in unsupported_tokens
        if tok not in DISCOURSE_TOKENS and tok not in INFERENCE_TOKENS
    ]
    biomedical_anchor_overlap = [
        tok for tok in overlap
        if any(ch.isdigit() for ch in tok) or len(tok) >= 5 or tok in INFERENCE_TOKENS
    ]
    return {
        "support_score": float(jacc),
        "token_precision": float(precision),
        "overlap_count": int(overlap_count),
        "exact_substring": exact_substring,
        "unsupported_tokens": unsupported_tokens,
        "discourse_unsupported": discourse_unsupported,
        "inference_unsupported": inference_unsupported,
        "novel_unsupported": novel_unsupported,
        "biomedical_anchor_overlap_count": len(biomedical_anchor_overlap),
    }


def _categorize_support(features: Dict[str, Any], has_kg: bool, claim_type: str = "", lenient: bool = False) -> str:
    strict_external = claim_type == "external_knowledge_claim" and not lenient

    if features["exact_substring"] and not strict_external:
        return "supported"
    if features["token_precision"] >= 0.85 and features["overlap_count"] >= 3 and not strict_external:
        return "supported"
    if (
        features["token_precision"] >= 0.70
        and features["support_score"] >= 0.45
        and not features["novel_unsupported"]
        and not strict_external
    ):
        return "supported"

    if (
        features["token_precision"] >= 0.55
        and features["overlap_count"] >= 3
        and len(features["novel_unsupported"]) == 0
        and not strict_external
    ):
        return "supported"

    if (
        has_kg
        and features["biomedical_anchor_overlap_count"] >= 2
        and features["token_precision"] >= 0.35
        and len(features["novel_unsupported"]) <= 1
        and not strict_external
    ):
        return "supported"

    if (
        features["biomedical_anchor_overlap_count"] >= 2
        and features["token_precision"] >= 0.35
        and len(features["novel_unsupported"]) == 0
    ):
        return "partial"

    if lenient and features["token_precision"] >= 0.30 and features["biomedical_anchor_overlap_count"] >= 1:
        return "partial"

    if lenient and features["support_score"] >= 0.20 and features["overlap_count"] >= 2:
        return "partial"

    if features["token_precision"] >= 0.45 or features["overlap_count"] >= 3 or features["support_score"] >= 0.25:
        return "partial"
    return "unsupported"


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _support_label_to_score(label: str) -> float:
    mapping = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    return float(mapping.get(label, 0.0))


def _to_ratio_summary(supported: int, partial: int, unsupported: int) -> Dict[str, float]:
    total = max(1, supported + partial + unsupported)
    return {
        "supported_claim_ratio": float(supported / total),
        "partial_support_ratio": float(partial / total),
        "unsupported_claim_ratio": float(unsupported / total),
    }


def _classify_claim_type(claim: str, field: str, normalized_explanation: Dict[str, Any], features: Dict[str, Any]) -> str:
    if field == "optional_inference":
        return "external_knowledge_claim"
    claim_tokens = set(tokenize(claim, drop_stopwords=False))
    if claim_tokens & CLINICAL_OUTCOME_HINTS:
        return "external_knowledge_claim"
    if claim_tokens & MECHANISM_HINTS and not features.get("exact_substring", False) and features.get("token_precision", 0.0) < 0.75:
        return "external_knowledge_claim"
    if features.get("exact_substring", False) or features.get("token_precision", 0.0) >= 0.85:
        return "extractive_claim"
    return "light_inferential_claim"


def _detect_hallucination_type(claim: str, evidence_text: str, query_pairs: List[str]) -> str:
    normalized_claim = normalize_text(claim)
    claim_tokens = set(tokenize(claim, drop_stopwords=False))
    evidence_tokens = set(tokenize(evidence_text, drop_stopwords=False))

    missing_clinical = (claim_tokens & CLINICAL_OUTCOME_HINTS) - evidence_tokens
    missing_mechanism = (claim_tokens & MECHANISM_HINTS) - evidence_tokens

    if missing_clinical:
        return "unsupported_clinical_outcome"
    if missing_mechanism:
        return "unsupported_specific_mechanism"
    if "->" in claim:
        normalized_pairs = {normalize_text(item) for item in query_pairs}
        if normalized_pairs and normalized_claim not in normalized_pairs:
            return "unsupported_pairing_statement"
    return "unsupported_claim"


def compute_faithfulness_for_sample(
    sentence: str,
    explanation: str,
    predicted_label: str,
    kg_evidence: str = "",
    queries: str = "",
) -> Dict[str, Any]:
    sentence_text = normalize_text(sentence)
    kg_text = normalize_text(kg_evidence)
    has_kg = bool(kg_text)
    evidence_text = normalize_text(f"{sentence_text} {kg_text}") if has_kg else sentence_text
    normalized_explanation = normalize_explanation_output(explanation, predicted_label=predicted_label)
    effective_label = normalize_label_text(normalized_explanation.get("predicted_label", ""))
    claim_records = extract_claim_records(explanation)
    query_coverage = compute_query_coverage(
        queries=queries,
        representative_pairs=normalized_explanation.get("representative_pairs", []),
    )
    claims = [item["claim"] for item in claim_records]

    label_notes: List[str] = []
    if normalized_explanation.get("label_backfilled"):
        label_notes.append(
            f"predicted_label was empty and was backfilled as {effective_label or 'unknown'} from explanation patterns"
        )
    if not effective_label:
        label_notes.append("predicted_label remains empty after explanation backfill")

    if not claims:
        return {
            "evidence_mode": "with_kg" if has_kg else "no_kg",
            "kg_used": has_kg,
            "coverage_ratio": 0.0,
            "hallucination_rate": 1.0,
            "partial_support_rate": 0.0,
            "consistency_score": 0.0,
            "kg_grounded_ratio": None,
            "supported_claims": [],
            "partial_supported_claims": [],
            "unsupported_claims": [],
            "optional_inference_claims": normalized_explanation.get("optional_inference", []),
            "unsupported_optional_inference_claims": [],
            "strict_sentence_grounding": _to_ratio_summary(0, 0, 0),
            "lenient_medical_plausibility": _to_ratio_summary(0, 0, 0),
            "hallucination_types": ["schema_noise"] if explanation and ("{" in explanation or "[" in explanation) else [],
            "query_coverage": query_coverage["query_coverage"],
            "covered_queries": query_coverage["covered_queries"],
            "total_queries": query_coverage["total_queries"],
            "evidence_sentence_coverage": float(1.0 if normalized_explanation.get("evidence_sent_ids") else 0.0),
            "predicted_label": effective_label,
            "label_backfilled": bool(normalized_explanation.get("label_backfilled", False)),
            "label_consistency_pass": False,
            "label_consistency_notes": label_notes,
            "normalized_explanation": normalized_explanation,
        }

    supported: List[Dict[str, Any]] = []
    partial_supported: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    optional_inference_claims: List[Dict[str, Any]] = []
    unsupported_optional_inference_claims: List[Dict[str, Any]] = []
    strict_counts = {"supported": 0, "partial": 0, "unsupported": 0}
    lenient_counts = {"supported": 0, "partial": 0, "unsupported": 0}
    hallucination_types: List[str] = []

    for item in claim_records:
        claim = item["claim"]
        features = _claim_support_features(claim, evidence_text)
        claim_type = _classify_claim_type(claim, item.get("field", ""), normalized_explanation, features)
        strict_category = _categorize_support(features, has_kg=has_kg, claim_type=claim_type, lenient=False)
        lenient_category = _categorize_support(features, has_kg=has_kg, claim_type=claim_type, lenient=True)
        record = {
            "claim": claim,
            "field": item.get("field", ""),
            "query": item.get("query", ""),
            "claim_type": claim_type,
            "support_label": "partially_supported" if strict_category == "partial" else strict_category,
            "support_score": _support_label_to_score(strict_category),
            "lenient_support_label": "partially_supported" if lenient_category == "partial" else lenient_category,
            "lenient_support_score": _support_label_to_score(lenient_category),
            "similarity_score": float(features["support_score"]),
            "token_precision": float(features["token_precision"]),
            "unsupported_tokens": list(features["unsupported_tokens"]),
            "novel_unsupported_tokens": list(features["novel_unsupported"]),
            "evidence_span": " ".join(tok for tok in tokenize(claim, drop_stopwords=False) if tok in set(tokenize(evidence_text, drop_stopwords=False)))[:200],
        }
        strict_counts[strict_category] += 1
        lenient_counts[lenient_category] += 1

        if strict_category == "supported":
            supported.append(record)
        elif strict_category == "partial":
            partial_supported.append(record)
        else:
            hallucination_types.append(_detect_hallucination_type(claim, evidence_text, query_coverage["query_pairs"]))
            unsupported.append(record)

    for item in normalized_explanation.get("optional_inference", []):
        claim = str(item.get("text", "")).strip()
        if not claim:
            continue
        features = _claim_support_features(claim, evidence_text)
        support_label = _categorize_support(
            features, has_kg=has_kg, claim_type="external_knowledge_claim", lenient=False
        )
        record = {
            "claim": claim,
            "claim_type": "external_knowledge_claim",
            "support_label": "partially_supported" if support_label == "partial" else support_label,
            "support_score": _support_label_to_score(support_label),
            "similarity_score": float(features["support_score"]),
            "token_precision": float(features["token_precision"]),
            "unsupported_tokens": list(features["unsupported_tokens"]),
        }
        optional_inference_claims.append(record)
        if support_label != "supported":
            unsupported_optional_inference_claims.append(record)
            hallucination_types.append(_detect_hallucination_type(claim, evidence_text, query_coverage["query_pairs"]))

    coverage_ratio = len(supported) / len(claims)
    hallucination_rate = len(unsupported) / len(claims)
    partial_support_rate = len(partial_supported) / len(claims)

    kg_grounded_ratio = None
    kg_grounded_claims: List[Dict[str, Any]] = []
    if has_kg:
        kg_anchors = _extract_kg_anchors(kg_evidence)
        kg_claim_records = list(claim_records)
        seen_kg_claims = {normalize_text(item["claim"]) for item in kg_claim_records}
        for item in normalized_explanation.get("optional_inference", []):
            claim = str(item.get("text", "") or "").strip()
            if claim and normalize_text(claim) not in seen_kg_claims:
                seen_kg_claims.add(normalize_text(claim))
                kg_claim_records.append({"claim": claim, "field": "optional_inference"})
        for item in kg_claim_records:
            matched_anchors = _claim_kg_anchor_overlap(item["claim"], kg_anchors)
            if matched_anchors:
                kg_grounded_claims.append({"claim": item["claim"], "matched_anchors": matched_anchors})
        kg_grounded_ratio = float(len(kg_grounded_claims) / len(kg_claim_records)) if kg_claim_records else 0.0

    label = effective_label
    label_terms = LABEL_KEYWORDS.get(label, set())
    exp_text = normalize_text(explanation)

    # If no label-specific terms are defined, fallback to neutral score.
    if not label_terms:
        consistency_score = 0.5
    else:
        hits = 0
        for term in label_terms:
            if term in exp_text:
                hits += 1
        consistency_score = min(1.0, hits / max(1, min(3, len(label_terms))))

    # Penalize explicit contradiction patterns.
    if label != "false" and "no interaction" in exp_text:
        consistency_score = max(0.0, consistency_score - 0.5)
    if label == "false" and any(k in exp_text for k in ("inhibition", "induction", "metabolism")):
        consistency_score = max(0.0, consistency_score - 0.4)

    label_consistency_pass = bool(label and consistency_score >= 0.5)
    if label and not label_consistency_pass:
        label_notes.append(f"predicted_label={label} is weakly supported by explanation patterns")

    return {
        "evidence_mode": "with_kg" if has_kg else "no_kg",
        "kg_used": has_kg,
        "coverage_ratio": float(coverage_ratio),
        "hallucination_rate": float(hallucination_rate),
        "partial_support_rate": float(partial_support_rate),
        "consistency_score": float(consistency_score),
        "kg_grounded_ratio": kg_grounded_ratio,
        "kg_grounded_claims": kg_grounded_claims,
        "supported_claims": supported,
        "partial_supported_claims": partial_supported,
        "unsupported_claims": unsupported,
        "optional_inference_claims": optional_inference_claims,
        "unsupported_optional_inference_claims": unsupported_optional_inference_claims,
        "strict_sentence_grounding": _to_ratio_summary(
            strict_counts["supported"], strict_counts["partial"], strict_counts["unsupported"]
        ),
        "lenient_medical_plausibility": _to_ratio_summary(
            lenient_counts["supported"], lenient_counts["partial"], lenient_counts["unsupported"]
        ),
        "hallucination_types": sorted(set(hallucination_types)),
        "query_coverage": query_coverage["query_coverage"],
        "covered_queries": query_coverage["covered_queries"],
        "total_queries": query_coverage["total_queries"],
        "query_coverage_denominator": query_coverage["query_coverage_denominator"],
        "representative_pair_precision": query_coverage["representative_pair_precision"],
        "evidence_sentence_coverage": float(1.0 if normalized_explanation.get("evidence_sent_ids") else 0.0),
        "predicted_label": effective_label,
        "predicted_label_source": normalized_explanation.get("label_source", ""),
        "label_backfilled": bool(normalized_explanation.get("label_backfilled", False)),
        "label_consistency_pass": label_consistency_pass,
        "label_consistency_notes": label_notes,
        "normalized_explanation": normalized_explanation,
    }


def aggregate_faithfulness(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "n_samples": 0,
            "coverage_mean": 0.0,
            "hallucination_mean": 0.0,
            "partial_support_mean": 0.0,
            "consistency_mean": 0.0,
            "kg_grounded_mean": None,
            "by_evidence_mode": {
                "with_kg": {"n_samples": 0, "coverage_mean": 0.0, "hallucination_mean": 0.0, "partial_support_mean": 0.0, "consistency_mean": 0.0, "kg_grounded_mean": None},
                "no_kg": {"n_samples": 0, "coverage_mean": 0.0, "hallucination_mean": 0.0, "partial_support_mean": 0.0, "consistency_mean": 0.0, "kg_grounded_mean": None},
            },
        }

    cov_values = [float(x.get("coverage_ratio", 0.0)) for x in samples]
    hal_values = [float(x.get("hallucination_rate", 0.0)) for x in samples]
    partial_values = [float(x.get("partial_support_rate", 0.0)) for x in samples]
    con_values = [float(x.get("consistency_score", 0.0)) for x in samples]
    kg_values = [float(x.get("kg_grounded_ratio")) for x in samples if x.get("kg_grounded_ratio") is not None]
    query_values = [float(x.get("query_coverage", 0.0)) for x in samples]
    strict_supported = [
        float((x.get("strict_sentence_grounding") or {}).get("supported_claim_ratio", 0.0))
        for x in samples
    ]
    strict_unsupported = [
        float((x.get("strict_sentence_grounding") or {}).get("unsupported_claim_ratio", 0.0))
        for x in samples
    ]
    lenient_supported = [
        float((x.get("lenient_medical_plausibility") or {}).get("supported_claim_ratio", 0.0))
        for x in samples
    ]

    cov = _mean(cov_values)
    hal = _mean(hal_values)
    partial = _mean(partial_values)
    con = _mean(con_values)

    mode_buckets: Dict[str, List[Dict[str, Any]]] = {"with_kg": [], "no_kg": []}
    for item in samples:
        mode = str(item.get("evidence_mode", "no_kg"))
        mode_buckets.setdefault(mode, []).append(item)

    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ("with_kg", "no_kg"):
        bucket = mode_buckets.get(mode, [])
        b_cov = [float(x.get("coverage_ratio", 0.0)) for x in bucket]
        b_hal = [float(x.get("hallucination_rate", 0.0)) for x in bucket]
        b_partial = [float(x.get("partial_support_rate", 0.0)) for x in bucket]
        b_con = [float(x.get("consistency_score", 0.0)) for x in bucket]
        b_kg = [float(x.get("kg_grounded_ratio")) for x in bucket if x.get("kg_grounded_ratio") is not None]
        by_mode[mode] = {
            "n_samples": len(bucket),
            "coverage_mean": _mean(b_cov),
            "hallucination_mean": _mean(b_hal),
            "partial_support_mean": _mean(b_partial),
            "consistency_mean": _mean(b_con),
            "kg_grounded_mean": _mean(b_kg) if b_kg else None,
        }

    return {
        "n_samples": len(samples),
        "coverage_mean": float(cov),
        "hallucination_mean": float(hal),
        "partial_support_mean": float(partial),
        "consistency_mean": float(con),
        "kg_grounded_mean": _mean(kg_values) if kg_values else None,
        "query_coverage_mean": _mean(query_values),
        "strict_supported_mean": _mean(strict_supported),
        "strict_unsupported_mean": _mean(strict_unsupported),
        "lenient_supported_mean": _mean(lenient_supported),
        "by_evidence_mode": by_mode,
    }
