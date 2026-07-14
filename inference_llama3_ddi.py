"""Inference and evaluation for LoRA-tuned Meta-Llama-3-8B-Instruct on DDI data.

Supports an optional per-sample agent loop (--enable_agent_loop): after each
generation the output is scored with `FaithfulnessCritic` (and optionally
`JudgeCritic`); if a critic fails, its feedback is appended to the messages and
the model regenerates. See docs/agent_loop.md for details.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Allow direct execution: python inference_llama3_ddi.py
_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from pipelines.query_utils import expand_query_pairs, get_query_group_text


PROMPT_MODES = {
    "auto",
    "full",
    "explanation_with_kg",
    "explanation_without_kg",
    "label_only_with_kg",
    "reasoning_without_kg",
    "label_only_without_kg",
}

PROMPT_PROFILES: Dict[str, Dict[str, Any]] = {
    "full": {
        "include_kg": True,
        "system_prompt": """You are an expert Clinical Pharmacologist and Graph Reasoning Assistant.
Given clinical text and KG evidence, provide a detailed analysis and output final DDI labels.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output must be a JSON list with fields:
query, analysis_steps, mechanism_summary, confidence_assessment, label.
""",
        "output_hint": "Use sentence + kg_evidence. Provide reasoning then final label.",
    },
    "explanation_with_kg": {
        "include_kg": True,
        "system_prompt": """You are an expert Clinical Pharmacologist and Graph Reasoning Assistant.
Given clinical text and KG evidence, provide detailed mechanism explanations only.

Output must be a JSON list with fields:
query, analysis_steps, mechanism_summary, confidence_assessment.
Do not output labels.
""",
        "output_hint": "Use sentence + kg_evidence. Provide reasoning only (no label).",
    },
    "label_only_with_kg": {
        "include_kg": True,
        "system_prompt": """You are an expert Clinical Pharmacologist.
Use sentence text and KG evidence to assign DDI labels.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output exactly one final DDI label only.
Do not output reasoning steps or JSON.
""",
    "output_hint": "Use sentence + kg_evidence and output one final label only (Mechanism/Effect/Advice/Int/False).",
    },
    "reasoning_without_kg": {
        "include_kg": False,
        "system_prompt": """You are an expert Clinical Pharmacologist.
Given sentence text only (no external KG evidence), provide detailed reasoning and final DDI labels.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output must be a JSON list with fields:
query, analysis_steps, mechanism_summary, confidence_assessment, label.
""",
        "output_hint": "No external KG. Use sentence only with detailed reasoning before final label.",
    },
    "explanation_without_kg": {
        "include_kg": False,
        "system_prompt": """You are an expert Clinical Pharmacologist.
Given sentence text only (no external KG evidence), provide detailed phenomenon-level explanations only.

Output must be a JSON list with fields:
query, analysis_steps, mechanism_summary, representative_pairs.
Do not output labels.
Set query to "GLOBAL_PHENOMENON" and keep representative_pairs to at most 3.
""",
        "output_hint": "No external KG. Use sentence only, output phenomenon-level reasoning without label, and compress evidence to <=3 representative pairs.",
    },
    "label_only_without_kg": {
        "include_kg": False,
        "system_prompt": """You are an expert Clinical Pharmacologist.
Given sentence text only (no external KG evidence), assign DDI labels.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output exactly one final DDI label only.
Do not output reasoning steps or JSON.
""",
    "output_hint": "No external KG. Output one final label only (Mechanism/Effect/Advice/Int/False).",
    },
}

VALID_LABELS = ["Mechanism", "Effect", "Advice", "Int", "False"]


def _default_max_new_tokens_for_mode(prompt_mode: str) -> int:
    if prompt_mode in {"label_only_with_kg", "label_only_without_kg"}:
        return 128
    if prompt_mode in {"explanation_with_kg", "explanation_without_kg"}:
        return 768
    # full / reasoning_without_kg
    return 512


def _infer_prompt_mode(adapter_dir: Path, input_path: Path, sample: Dict[str, Any]) -> str:
    names = f"{adapter_dir.name.lower()}::{input_path.name.lower()}"
    if "label_only_without_kg" in names:
        return "label_only_without_kg"
    if "explanation_without_kg" in names:
        return "explanation_without_kg"
    if "reasoning_without_kg" in names:
        return "reasoning_without_kg"
    if "explanation_with_kg" in names:
        return "explanation_with_kg"
    if "label_only_with_kg" in names:
        return "label_only_with_kg"

    inp = sample.get("input", {}) if isinstance(sample, dict) else {}
    has_kg = bool(str(inp.get("kg_evidence", "") or "").strip())
    output_text = str(sample.get("output", "") or "")
    has_reasoning = "analysis_steps" in output_text or "mechanism_summary" in output_text
    has_label = '"label"' in output_text or "label" in output_text.lower()

    if has_kg:
        if has_reasoning:
            return "full" if has_label else "explanation_with_kg"
        return "label_only_with_kg"
    if has_reasoning:
        return "reasoning_without_kg" if has_label else "explanation_without_kg"
    return "label_only_without_kg"


def _resolve_prompt_mode(requested_mode: str, adapter_dir: Path, input_path: Path, sample: Dict[str, Any]) -> str:
    alias = {
        "classification_with_kg": "label_only_with_kg",
        "classification_without_kg": "label_only_without_kg",
        "explain_with_kg": "explanation_with_kg",
        "explain_without_kg": "explanation_without_kg",
        "with_kg_label_only": "label_only_with_kg",
        "without_kg_label_only": "label_only_without_kg",
        "without_kg_reasoning": "reasoning_without_kg",
        "reasoning_with_kg": "full",
    }
    mode = alias.get(requested_mode, requested_mode)
    if mode == "auto":
        mode = _infer_prompt_mode(adapter_dir=adapter_dir, input_path=input_path, sample=sample)
    if mode not in PROMPT_PROFILES:
        raise ValueError(f"Unsupported prompt mode: {requested_mode}")
    return mode


def _truncate_text(text: Optional[str], max_chars: int) -> str:
    s = str(text or "")
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip() + " ...[TRUNCATED]"
    return s


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _cleanup_cuda() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    mapping = {
        "mechanism": "Mechanism",
        "effect": "Effect",
        "advice": "Advice",
        "int": "Int",
        "false": "False",
    }
    return mapping.get(s)


def _extract_labels_from_output(output_text: str) -> List[str]:
    labels: List[str] = []
    text = (output_text or "").strip()
    if not text:
        return labels

    # Try strict JSON parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    norm = _normalize_label(item.get("label"))
                    if norm:
                        labels.append(norm)
            if labels:
                return labels
    except Exception:
        pass

    # Fallback: regex for JSON-like label fields.
    for m in re.finditer(r'"label"\s*:\s*"(Mechanism|Effect|Advice|Int|False)"', text, flags=re.IGNORECASE):
        norm = _normalize_label(m.group(1))
        if norm:
            labels.append(norm)

    # Fallback: plain text label-only output.
    if not labels:
        plain = _normalize_label(text)
        if plain:
            labels.append(plain)
    return labels


def _majority_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _extract_query_list(entry: Dict[str, Any]) -> List[str]:
    return expand_query_pairs(get_query_group_text(entry.get("input", {})))


def _normalize_reasoning_output(
    decoded: str,
    entry: Dict[str, Any],
    fallback_label: Optional[str],
    include_label: bool,
) -> str:
    """Normalize reasoning-mode output to JSON list string aligned with source dataset schema."""
    text = str(decoded or "").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, list) and parsed:
        normalized_list = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "") or "").strip()
            if not query:
                continue
            normalized_item = {
                "query": query,
                "analysis_steps": str(item.get("analysis_steps", "") or ""),
                "mechanism_summary": str(item.get("mechanism_summary", "") or ""),
                "confidence_assessment": str(item.get("confidence_assessment", "") or ""),
            }
            if include_label:
                normalized_item["label"] = _normalize_label(item.get("label")) or fallback_label or "Int"
            normalized_list.append(normalized_item)
        if normalized_list:
            return json.dumps(normalized_list, ensure_ascii=False, indent=4)

    query_list = _extract_query_list(entry)
    if not query_list:
        query_list = ["UNKNOWN_QUERY"]
    label = fallback_label or "Int"
    repaired = []
    for q in query_list:
        item = {
            "query": q,
            "analysis_steps": "",
            "mechanism_summary": "",
            "confidence_assessment": "",
        }
        if include_label:
            item["label"] = label
        repaired.append(item)
    return json.dumps(repaired, ensure_ascii=False, indent=4)


def _strip_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _parse_json_list_tolerant(text: str) -> Optional[List[Dict[str, Any]]]:
    s = _strip_code_fence(text)
    if not s:
        return None

    candidates = [s]
    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        candidates.append(s[l : r + 1])

    tried = set()
    for cand in candidates:
        if cand in tried:
            continue
        tried.add(cand)

        variants = [cand]
        if cand.count("{") == cand.count("}") + 1 and cand.rstrip().endswith("]"):
            idx = cand.rfind("]")
            variants.append(cand[:idx] + "}" + cand[idx:])

        for var in variants:
            try:
                parsed = json.loads(var)
            except Exception:
                continue
            if isinstance(parsed, list):
                dict_items = [x for x in parsed if isinstance(x, dict)]
                if dict_items:
                    return dict_items
    return None


def _fallback_representative_pairs(entry: Dict[str, Any], max_items: int = 3) -> List[str]:
    pairs = []
    for q in _extract_query_list(entry):
        if q and q not in pairs:
            pairs.append(q)
        if len(pairs) >= max_items:
            break
    return pairs


def _normalize_explanation_output(decoded: str, entry: Dict[str, Any]) -> str:
    """Normalize explanation-mode output to GLOBAL_PHENOMENON schema."""
    parsed = _parse_json_list_tolerant(decoded)
    if parsed:
        item = parsed[0]
        rep = item.get("representative_pairs")
        if isinstance(rep, list):
            representative_pairs = [str(x).strip() for x in rep if str(x).strip()][:3]
        else:
            representative_pairs = []
        if not representative_pairs:
            representative_pairs = _fallback_representative_pairs(entry, max_items=3)

        normalized = {
            "query": str(item.get("query") or "GLOBAL_PHENOMENON"),
            "analysis_steps": str(item.get("analysis_steps") or ""),
            "mechanism_summary": str(item.get("mechanism_summary") or ""),
            "representative_pairs": representative_pairs,
        }
        return json.dumps([normalized], ensure_ascii=False, indent=4)

    # Regex salvage for near-JSON outputs.
    text = _strip_code_fence(decoded)
    analysis = ""
    mechanism = ""
    m = re.search(r'"analysis_steps"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
    if m:
        analysis = m.group(1)
    m = re.search(r'"mechanism_summary"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
    if m:
        mechanism = m.group(1)

    found_pairs = []
    for pm in re.finditer(r"([^\n\",\[\]]+\s*->\s*[^\n\",\[\]]+)", text):
        p = pm.group(1).strip()
        if p and p not in found_pairs:
            found_pairs.append(p)
        if len(found_pairs) >= 3:
            break
    if not found_pairs:
        found_pairs = _fallback_representative_pairs(entry, max_items=3)

    repaired = {
        "query": "GLOBAL_PHENOMENON",
        "analysis_steps": analysis,
        "mechanism_summary": mechanism,
        "representative_pairs": found_pairs,
    }
    return json.dumps([repaired], ensure_ascii=False, indent=4)


def _align_output_to_mode(
    decoded: str,
    prompt_mode: str,
    entry: Dict[str, Any],
    pred_label: Optional[str],
) -> str:
    """Align output field format with the corresponding finetune source dataset."""
    if prompt_mode in {"label_only_with_kg", "label_only_without_kg"}:
        return pred_label or "Int"
    if prompt_mode in {"explanation_with_kg", "explanation_without_kg"}:
        return _normalize_explanation_output(decoded=decoded, entry=entry)
    include_label = prompt_mode in {"full", "reasoning_without_kg"}
    return _normalize_reasoning_output(
        decoded=decoded,
        entry=entry,
        fallback_label=pred_label,
        include_label=include_label,
    )


def _compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    labels = VALID_LABELS
    support = {lb: 0 for lb in labels}
    tp = {lb: 0 for lb in labels}
    fp = {lb: 0 for lb in labels}
    fn = {lb: 0 for lb in labels}

    for g, p in zip(gold, pred):
        if g in support:
            support[g] += 1
        if g == p and g in tp:
            tp[g] += 1
        else:
            if p in fp:
                fp[p] += 1
            if g in fn:
                fn[g] += 1

    per_class = {}
    f1_list = []
    weighted_f1_num = 0.0
    total = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)

    for lb in labels:
        p_den = tp[lb] + fp[lb]
        r_den = tp[lb] + fn[lb]
        prec = tp[lb] / p_den if p_den > 0 else 0.0
        rec = tp[lb] / r_den if r_den > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[lb] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support[lb],
        }
        f1_list.append(f1)
        weighted_f1_num += f1 * support[lb]

    metrics = {
        "n_samples": total,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "macro_f1": (sum(f1_list) / len(f1_list)) if f1_list else 0.0,
        "weighted_f1": (weighted_f1_num / total) if total > 0 else 0.0,
        "per_class": per_class,
    }
    return metrics


def build_messages(
    entry: Dict[str, Any],
    prompt_mode: str,
    max_kg_evidence_chars: int = 0,
    max_sentence_chars: int = 0,
    max_queries_chars: int = 0,
) -> List[Dict[str, str]]:
    instruction = entry.get("instruction", "Analyze the biological mechanisms based on KG evidence.")
    input_data = entry.get("input", {})
    profile = PROMPT_PROFILES[prompt_mode]

    sentence = _truncate_text(input_data.get("sentence", ""), max_sentence_chars)
    kg_raw = str(input_data.get("kg_evidence", "") or "")
    kg_evidence = _truncate_text(kg_raw, max_kg_evidence_chars)
    query_group = _truncate_text(get_query_group_text(input_data), max_queries_chars)

    payload = {
        "sentence": sentence,
        "query_group": query_group,
    }
    if profile["include_kg"]:
        payload["kg_evidence"] = kg_evidence
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)

    current_user = f"""Instruction: {instruction}

Input Data:
{input_json}

Prompt Strategy: {prompt_mode}
{profile["output_hint"]}

Output:"""

    return [
        {"role": "system", "content": profile["system_prompt"]},
        {"role": "user", "content": current_user},
    ]


def load_model_and_tokenizer(base_model: str, adapter_dir: str, use_multi_gpu: bool) -> (Any, Any):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    from_pretrained_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        max_memory = {
            i: int(torch.cuda.get_device_properties(i).total_memory * 0.9)
            for i in range(torch.cuda.device_count())
        }
        from_pretrained_kwargs["device_map"] = "balanced"
        from_pretrained_kwargs["max_memory"] = max_memory
        pretty = {f"cuda:{k}": v for k, v in max_memory.items()}
        print(f"Using single-process multi-GPU sharding: {pretty}")
    else:
        from_pretrained_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(base_model, **from_pretrained_kwargs)

    # Allow no-LoRA baseline runs with the same inference stack/prompt template.
    adapter_raw = str(adapter_dir or "").strip()
    adapter_path = Path(adapter_raw) if adapter_raw else None
    skip_adapter = (not adapter_raw) or adapter_raw.lower() in {"none", "null", "no_lora", "base"}
    if not skip_adapter and adapter_path is not None and adapter_path.exists():
        model = PeftModel.from_pretrained(model, adapter_raw)
        print(f"Loaded LoRA adapter from: {adapter_raw}")
    else:
        print("Running without LoRA adapter (base model only).")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_inference(args: argparse.Namespace):
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    adapter_path = Path(args.adapter_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.load(open(input_path, "r", encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    sample = data[0] if data else {}
    prompt_mode = _resolve_prompt_mode(
        requested_mode=args.prompt_mode,
        adapter_dir=adapter_path,
        input_path=input_path,
        sample=sample,
    )
    print(f"Prompt engineering mode: {prompt_mode}")

    effective_max_new_tokens = int(args.max_new_tokens)
    if effective_max_new_tokens <= 0:
        effective_max_new_tokens = _default_max_new_tokens_for_mode(prompt_mode)
        print(f"Auto max_new_tokens by prompt_mode: {effective_max_new_tokens}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_dir, args.multi_gpu_shard)
    if hasattr(model, "config"):
        try:
            model.config.use_cache = True
        except Exception:
            pass

    sidecar_labels = None
    if args.labels_json:
        with open(args.labels_json, "r", encoding="utf-8") as f:
            sidecar_labels = json.load(f)

    results = []
    gold_list: List[str] = []
    pred_list: List[str] = []
    failed_indices: List[int] = []

    def _candidate_retry_plans() -> List[Dict[str, int]]:
        # Plan-0 keeps user settings intact. Later plans are only used if OOM occurs.
        plans = [
            {
                "max_new_tokens": int(effective_max_new_tokens),
                "max_kg_evidence_chars": int(args.max_kg_evidence_chars),
                "max_sentence_chars": int(args.max_sentence_chars),
                "max_queries_chars": int(args.max_queries_chars),
            },
            {
                "max_new_tokens": min(int(effective_max_new_tokens), 1024),
                "max_kg_evidence_chars": int(args.max_kg_evidence_chars) or 20000,
                "max_sentence_chars": int(args.max_sentence_chars),
                "max_queries_chars": int(args.max_queries_chars) or 4000,
            },
            {
                "max_new_tokens": min(int(effective_max_new_tokens), 768),
                "max_kg_evidence_chars": 12000,
                "max_sentence_chars": int(args.max_sentence_chars) or 512,
                "max_queries_chars": 2500,
            },
            {
                "max_new_tokens": min(int(effective_max_new_tokens), 512),
                "max_kg_evidence_chars": 8000,
                "max_sentence_chars": int(args.max_sentence_chars) or 512,
                "max_queries_chars": 1800,
            },
            {
                "max_new_tokens": min(int(effective_max_new_tokens), 256),
                "max_kg_evidence_chars": 4000,
                "max_sentence_chars": int(args.max_sentence_chars) or 384,
                "max_queries_chars": 1200,
            },
        ]

        uniq: List[Dict[str, int]] = []
        seen = set()
        for p in plans:
            key = (
                p["max_new_tokens"],
                p["max_kg_evidence_chars"],
                p["max_sentence_chars"],
                p["max_queries_chars"],
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    retry_plans = _candidate_retry_plans()

    # --------------------------------------------------------------------
    # 抽出"一条 messages -> decoded"的核心生成，包含 OOM 逐级降级。供普通推理
    # 与 agent-loop 两条路径复用。
    # --------------------------------------------------------------------
    def _generate_from_messages(messages: List[Dict[str, str]], sample_idx: int) -> str:
        decoded_local = ""
        for attempt_id, plan in enumerate(retry_plans, 1):
            inputs = None
            gen = None
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=plan["max_new_tokens"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.temperature > 0,
                    )
                decoded_local = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True).strip()
                return decoded_local
            except Exception as e:
                if _is_oom_error(e):
                    print(
                        f"[OOM] sample={sample_idx} attempt={attempt_id}/{len(retry_plans)} "
                        f"plan(new={plan['max_new_tokens']},kg={plan['max_kg_evidence_chars']},q={plan['max_queries_chars']})"
                    )
                    _cleanup_cuda()
                    continue
                raise
            finally:
                if gen is not None:
                    del gen
                if inputs is not None:
                    del inputs
        return decoded_local

    # --- 可选 Agent Loop 初始化 ---
    agent_loop = None
    agent_traces_by_id: Dict[str, Any] = {}
    if getattr(args, "enable_agent_loop", False):
        from pipelines.agent_loop import (
            AgentLoopReasoner,
            AgentLoopConfig,
            FaithfulnessCritic,
            JudgeCritic,
        )
        critics = [
            FaithfulnessCritic(
                min_coverage=args.faith_min_coverage,
                max_hallucination=args.faith_max_hallucination,
            )
        ]
        if args.use_judge_critic:
            api_key = args.judge_api_key or os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise ValueError("--use_judge_critic requires --judge_api_key or DASHSCOPE_API_KEY env var")
            critics.append(
                JudgeCritic(
                    api_key=api_key,
                    model_id=args.judge_model_id,
                    base_url=args.judge_base_url,
                    min_overall_score=args.judge_min_overall_score,
                )
            )
        agent_loop = AgentLoopReasoner(
            reasoner=None,  # 每样本重新绑定
            critics=critics,
            config=AgentLoopConfig(max_rounds=args.agent_max_rounds, verbose=True),
        )
        print(
            f"[AgentLoop] enabled: max_rounds={args.agent_max_rounds}, "
            f"faith(cov>={args.faith_min_coverage},hall<={args.faith_max_hallucination}), "
            f"judge={'on(min='+str(args.judge_min_overall_score)+')' if args.use_judge_critic else 'off'}"
        )

    for idx, entry in enumerate(data, 1):
        decoded = ""
        pred_label = None
        agent_trace_dict: Optional[Dict[str, Any]] = None

        base_plan = retry_plans[0]
        initial_messages = build_messages(
            entry,
            prompt_mode=prompt_mode,
            max_kg_evidence_chars=base_plan["max_kg_evidence_chars"],
            max_sentence_chars=base_plan["max_sentence_chars"],
            max_queries_chars=base_plan["max_queries_chars"],
        )

        if agent_loop is not None:
            # 通过闭包给 loop 提供 reasoner
            def _reasoner(messages, round_idx, _entry=entry, _idx=idx):
                return _generate_from_messages(messages, sample_idx=_idx)

            agent_loop.reasoner = _reasoner
            input_data_view = entry.get("input", {}) if isinstance(entry, dict) else {}
            sample_view = {
                "sentence": str(input_data_view.get("sentence", "")),
                "query_group": get_query_group_text(input_data_view),
                "kg_evidence": str(input_data_view.get("kg_evidence", "")),
                "predicted_label": "",
                "gold_label": str(input_data_view.get("gold_label", "")),
            }
            trace = agent_loop.run(sample=sample_view, initial_messages=initial_messages)
            decoded = trace.final_output
            from pipelines.agent_loop import trace_to_dict
            agent_trace_dict = trace_to_dict(trace, keep_full_messages=False)
            agent_traces_by_id[str(idx - 1)] = trace
        else:
            decoded = _generate_from_messages(initial_messages, sample_idx=idx)

        if prompt_mode in {"explanation_with_kg", "explanation_without_kg"}:
            pred_label = None
        else:
            labels = _extract_labels_from_output(decoded)
            pred_label = _majority_label(labels)

        if pred_label is None and not decoded:
            failed_indices.append(idx - 1)
            _cleanup_cuda()

        out_entry = dict(entry)
        aligned_output = _align_output_to_mode(
            decoded=decoded,
            prompt_mode=prompt_mode,
            entry=entry,
            pred_label=pred_label,
        )

        # For reasoning modes, enforce a non-null predicted label by re-parsing
        # the aligned output that already contains a repaired label field.
        if prompt_mode in {"full", "reasoning_without_kg"} and pred_label is None:
            labels = _extract_labels_from_output(aligned_output)
            pred_label = _majority_label(labels)

        out_entry["output"] = aligned_output
        out_entry["predicted_label"] = pred_label or ""
        if agent_trace_dict is not None:
            out_entry["agent_loop"] = {
                "n_rounds": agent_trace_dict["n_rounds"],
                "final_passed": agent_trace_dict["final_passed"],
                "final_score": agent_trace_dict["final_score"],
                "stopped_reason": agent_trace_dict["stopped_reason"],
            }

        gold_label = None
        if sidecar_labels is not None and idx - 1 < len(sidecar_labels):
            gold_label = _normalize_label(sidecar_labels[idx - 1].get("gold_label"))
        if gold_label is None:
            input_data = out_entry.get("input", {}) if isinstance(out_entry, dict) else {}
            gold_label = _normalize_label(input_data.get("gold_label")) or _normalize_label(input_data.get("target_label"))

        out_entry["gold_label"] = gold_label or ""
        out_entry["true_label"] = gold_label or ""

        if gold_label in VALID_LABELS and pred_label in VALID_LABELS:
                gold_list.append(gold_label)
                pred_list.append(pred_label)

        results.append(out_entry)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(data)}")
            _cleanup_cuda()

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved predictions to {output_path}")

    # Agent-loop 全量 trace 落到同目录，便于观察反思轨迹
    if agent_traces_by_id:
        from pipelines.agent_loop import save_trace_batch
        trace_path = output_path.parent / f"agent_trace_{output_path.stem}.json"
        save_trace_batch(agent_traces_by_id, str(trace_path), keep_full_messages=False)
        print(f"Saved agent-loop trace to {trace_path}")

    if sidecar_labels is not None:
        metrics = _compute_metrics(gold_list, pred_list)
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {metrics_path}")
        print(
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"Weighted-F1={metrics['weighted_f1']:.4f}, "
            f"Evaluated={metrics['n_samples']}"
        )
    if failed_indices:
        print(f"Warning: {len(failed_indices)} samples produced empty outputs after OOM retries.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with LoRA-tuned Llama3 DDI model")
    parser.add_argument("--base_model", type=str, default="models/Meta-Llama-3-8B-Instruct", help="Base model path or hub id")
    parser.add_argument("--adapter_dir", type=str, default="results/llama3_ddi_lora", help="LoRA adapter directory")
    parser.add_argument("--input_json", type=str, default="data/finetune/test/input_test.json", help="Input JSON file (under data/finetune/test/)")
    parser.add_argument("--labels_json", type=str, default="data/finetune/test/input_test_labels.json", help="Gold labels sidecar JSON for evaluation (under data/finetune/test/)")
    parser.add_argument("--output_json", type=str, default="data/reports/default/inference/test_out.json", help="Output JSON file (under data/reports/{exp}/inference/)")
    parser.add_argument("--metrics_json", type=str, default="data/reports/default/inference/test_metrics.json", help="Output metrics JSON file (under data/reports/{exp}/inference/)")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=-1,
        help="Generation length; <=0 means auto by prompt mode (label=128, reasoning=512, explanation=768)",
    )
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--max_kg_evidence_chars", type=int, default=0, help="Optional cap for kg_evidence chars (0=no cap)")
    parser.add_argument("--max_sentence_chars", type=int, default=0, help="Optional cap for sentence chars (0=no cap)")
    parser.add_argument("--max_queries_chars", type=int, default=0, help="Optional cap for queries chars (0=no cap)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--multi_gpu_shard", action="store_true", help="Shard base model across multiple GPUs in one process")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="auto",
        choices=sorted(PROMPT_MODES),
        help="Prompt engineering mode: auto/full/explanation_with_kg/explanation_without_kg/label_only_with_kg/reasoning_without_kg/label_only_without_kg",
    )
    # --- Agent Loop (self-critique-retry) options ---
    parser.add_argument("--enable_agent_loop", action="store_true",
                        help="Per-sample self-critique-retry loop (faithfulness + optional judge)")
    parser.add_argument("--agent_max_rounds", type=int, default=3,
                        help="Total rounds including the initial generation (default: 3)")
    parser.add_argument("--faith_min_coverage", type=float, default=0.4,
                        help="Faithfulness coverage_ratio threshold for pass (default: 0.4)")
    parser.add_argument("--faith_max_hallucination", type=float, default=0.5,
                        help="Faithfulness hallucination_rate ceiling for pass (default: 0.5)")
    parser.add_argument("--use_judge_critic", action="store_true",
                        help="Also use qwen-max judge as critic (expensive; needs DASHSCOPE_API_KEY)")
    parser.add_argument("--judge_min_overall_score", type=float, default=7.0,
                        help="Judge mechanism_overall_score threshold for pass (0-12, default: 7)")
    parser.add_argument("--judge_model_id", type=str, default="qwen-max")
    parser.add_argument("--judge_base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--judge_api_key", type=str, default="",
                        help="Judge API key; falls back to DASHSCOPE_API_KEY env var")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
