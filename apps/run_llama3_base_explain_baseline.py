import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipelines.query_utils import get_query_group_text

def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[TRUNCATED]"


def _build_messages(entry: Dict[str, Any], max_queries_chars: int) -> List[Dict[str, str]]:
    input_data = entry.get("input", {}) if isinstance(entry, dict) else {}
    sentence = str(input_data.get("sentence", "") or "")
    query_group = _truncate(get_query_group_text(input_data), max_queries_chars)
    predicted_label = str(input_data.get("predicted_label", "") or "").strip()

    system_prompt = """You are an expert Clinical Pharmacologist.
Given clinical sentence and drug-pair queries, provide explainable DDI reasoning.

Output must be a JSON list. For each query include fields:
query, analysis_steps, mechanism_summary.
Do not output labels.
"""

    label_hint = ""
    if predicted_label:
        label_hint = (
            f"Target relation label to explain: {predicted_label}.\\n"
            "Use this predicted_label as the only explanation target.\\n"
            "Ignore any gold label if present.\\n\\n"
        )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Query group:\n{query_group}\n\n"
        f"{label_hint}"
        "Return only a JSON list; do not add extra prose."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-finetuned Llama3 explainable-reasoning baseline")
    parser.add_argument("--base_model", type=str, default="models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--input_json", type=str, default="data/reports/_legacy_test_predictions/finetune_dataset_input_test_out_label_only_without_kg.json")
    parser.add_argument("--output_json", type=str, default="data/reports/default/inference/llama3_base_explanation_no_kg_no_lora.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_queries_chars", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _resolve_target_label(item: Dict[str, Any]) -> str:
    for key in ("predicted_label", "output", "gold_label"):
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _normalize_label(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    mapping = {
        "mechanism": "Mechanism",
        "effect": "Effect",
        "advice": "Advice",
        "int": "Int",
        "false": "False",
    }
    lower = s.lower()
    return mapping.get(lower, s)


def _drop_confidence_fields(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_drop_confidence_fields(x) for x in obj]
    if isinstance(obj, dict):
        return {
            k: _drop_confidence_fields(v)
            for k, v in obj.items()
            if k != "confidence_assessment"
        }
    return obj


def _sanitize_decoded_output(decoded: str) -> str:
    text = str(decoded or "").strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
        cleaned = _drop_confidence_fields(parsed)
        return json.dumps(cleaned, ensure_ascii=False, indent=2)
    except Exception:
        # Best-effort fallback when model returns non-strict JSON.
        return text.replace('"confidence_assessment":', '"removed_confidence_assessment":')


def _to_explain_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    input_data = item.get("input", {}) if isinstance(item, dict) else {}
    predicted_label = _normalize_label(item.get("predicted_label"))
    if not predicted_label:
        predicted_label = _normalize_label(_resolve_target_label(item))
    gold_label = _normalize_label(item.get("gold_label"))
    return {
        "instruction": "Explain why each queried drug pair is assigned the target DDI label using sentence-only evidence.",
        "input": {
            "sentence": str(input_data.get("sentence", "") or ""),
            "query_group": get_query_group_text(input_data),
            "predicted_label": predicted_label,
            "gold_label": gold_label,
        },
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    data = [_to_explain_sample(item) for item in source_data]

    if args.limit:
        data = data[: args.limit]

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: List[Dict[str, Any]] = []

    retry_plans = [
        {"new": int(args.max_new_tokens), "q": int(args.max_queries_chars)},
        {"new": min(int(args.max_new_tokens), 384), "q": 1800},
        {"new": min(int(args.max_new_tokens), 256), "q": 1400},
    ]

    for idx, entry in enumerate(data, 1):
        decoded = ""

        for plan_idx, plan in enumerate(retry_plans, 1):
            inputs = None
            gen = None
            try:
                messages = _build_messages(entry, max_queries_chars=plan["q"])
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                in_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=plan["new"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.temperature > 0,
                    )
                decoded = tokenizer.decode(gen[0][in_len:], skip_special_tokens=True).strip()
                decoded = _sanitize_decoded_output(decoded)
                break
            except Exception as e:
                if _is_oom_error(e):
                    print(f"[OOM] sample={idx}/{len(data)} plan={plan_idx}/{len(retry_plans)}")
                    _cleanup_cuda()
                    continue
                raise
            finally:
                if gen is not None:
                    del gen
                if inputs is not None:
                    del inputs

        src_input = entry.get("input", {}) if isinstance(entry, dict) else {}
        out = {
            "instruction": "Explain DDI mechanisms for the given sentence and query_group without external KG evidence.",
            "input": {
                "sentence": str(src_input.get("sentence", "") or ""),
                "query_group": get_query_group_text(src_input),
                "predicted_label": str(src_input.get("predicted_label", "") or ""),
                "gold_label": str(src_input.get("gold_label", "") or ""),
            },
            "output": decoded,
        }
        results.append(out)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(data)}")
            _cleanup_cuda()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved baseline predictions to {output_path}")


if __name__ == "__main__":
    main()
