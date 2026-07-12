"""
使用通用 LLM 接口为微调数据集补充输出（蒸馏生成）。
读取已构造的输入数据，按与 generate_outputs.py 一致的提示词和消息结构逐条生成。
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.io_utils import load_config
from pipelines.query_utils import get_query_group_text

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_SAVE_EVERY = 100
DEFAULT_LOG_EVERY = 10
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_REQUEST_RETRIES = 4
DEFAULT_REQUEST_RETRY_DELAY = 2.0

SYSTEM_PROMPT_WITH_KG = """You are an expert Clinical Pharmacologist. Classify Drug-Drug Interactions (DDI) from text and KG evidence into 1 of 5 categories:
1. Mechanism: PK mechanisms (CYP inhibition, absorption).
2. Effect: PD effects or clinical symptoms.
3. Advice: Medical recommendations/monitoring.
4. Int: Interaction stated, but no mechanism/effect details.
5. False: No interaction described for the specific pair.

### CRITICAL GUARDRAILS
1. TARGET FOCUS: ONLY analyze the exact drug pairs in 'query_group'. IGNORE other drugs. If text doesn't describe an interaction for YOUR pair, classify as 'False'.
2. NO HALLUCINATION: Rely strictly on text and provided kg_evidence. Do not invent pathways.

### OUTPUT FORMAT
Output a JSON list of objects:
[{"query": "A -> B", "analysis_steps": "Step 1... 2... Classification: <logic>", "mechanism_summary": "<brief>", "confidence_assessment": "High/Medium/Low", "label": "<Category>"}]"""

SYSTEM_PROMPT_NO_KG = """You are an expert Clinical Pharmacologist. Classify Drug-Drug Interactions (DDI) from sentence text into 1 of 5 categories:
1. Mechanism: PK mechanisms (CYP inhibition, absorption).
2. Effect: PD effects or clinical symptoms.
3. Advice: Medical recommendations/monitoring.
4. Int: Interaction stated, but no mechanism/effect details.
5. False: No interaction described for the specific pair.

### CRITICAL GUARDRAILS
1. TARGET FOCUS: ONLY analyze the exact drug pairs in 'query_group'. IGNORE other drugs. If text doesn't describe an interaction for YOUR pair, classify as 'False'.
2. NO HALLUCINATION: Rely strictly on the sentence text. Do not invent pathways.

### OUTPUT FORMAT
Output a JSON list of objects:
[{"query": "A -> B", "analysis_steps": "Step 1... 2... Classification: <logic>", "mechanism_summary": "<brief>", "confidence_assessment": "High/Medium/Low", "label": "<Category>"}]"""

FEW_SHOT_USER = """Instruction: Analyze the biological mechanisms for the specific drug pairs.

Input Data: { "sentence": "Absorption of tetracycline is impaired by bismuth subsalicylate.", "query_group": "tetracycline -> bismuth subsalicylate", "kg_evidence": "Mechanism focus: absorption.\\n\\nReliable KG anchors:\\n- tetracycline: ALB" }
"""

FEW_SHOT_ASSISTANT = """[ 
    { 
        "query": "tetracycline -> bismuth subsalicylate", 
        "analysis_steps": "1. Target Focus: Analyzing 'tetracycline -> bismuth subsalicylate'.\n2. Relevance Check: Sentence describes impaired absorption between these exact two drugs.\n3. KG Evidence Check: Shared node is ALB(Carrier), a non-specific transport protein.\n4. Synthesis: Bismuth subsalicylate physically interacts with tetracycline in the gut, reducing its systemic absorption.\n5. Classification Decision: The text primarily describes an alteration in 'absorption', which is a fundamental pharmacokinetic process. Thus, it is classified as 'Mechanism'.", 
        "mechanism_summary": "Physicochemical interaction: Bismuth subsalicylate likely chelates Tetracycline in the gastrointestinal tract, reducing its absorption.", 
        "confidence_assessment": "Medium",
        "label": "Mechanism"
    } 
]
"""

FEW_SHOT_USER_NO_KG = """Instruction: Analyze the biological mechanisms for the specific drug pairs.

Input Data: { "sentence": "Absorption of tetracycline is impaired by bismuth subsalicylate.", "query_group": "tetracycline -> bismuth subsalicylate" }
"""

FEW_SHOT_ASSISTANT_NO_KG = """[ 
    { 
        "query": "tetracycline -> bismuth subsalicylate", 
        "analysis_steps": "1. Target Focus: Analyzing 'tetracycline -> bismuth subsalicylate'.\n2. Relevance Check: Sentence explicitly states absorption is impaired between these exact two drugs.\n3. Text-grounded Interpretation: The interaction is an absorption change, which indicates a pharmacokinetic mechanism.\n4. Classification Decision: Classify as 'Mechanism' based on text evidence.", 
        "mechanism_summary": "The sentence indicates reduced tetracycline absorption when co-administered with bismuth subsalicylate.", 
        "confidence_assessment": "High",
        "label": "Mechanism"
    } 
]
"""

SYSTEM_PROMPT_EXPLANATION_WITH_KG = """You are an expert Clinical Pharmacologist and Graph Reasoning Assistant.
Generate explanation-only outputs from sentence text and KG evidence.

### OUTPUT FORMAT
Output a JSON list of objects:
[{"query": "A -> B", "analysis_steps": "Step 1... 2...", "mechanism_summary": "<brief>", "confidence_assessment": "High/Medium/Low"}]
Do not output any label field.
"""

SYSTEM_PROMPT_EXPLANATION_NO_KG = """You are an expert Clinical Pharmacologist.
Generate concise, evaluation-friendly explanation-only outputs from sentence text only.

### OUTPUT FORMAT
Output a JSON list of objects:
[{"query": "GLOBAL_PHENOMENON", "analysis_steps": "Step 1... 2...", "mechanism_summary": "<brief>", "representative_pairs": ["A -> B", "C -> D"]}]
Do not output any label field, confidence field, or uncertainty score.

### GENERATION RULES
1. Only use facts explicitly supported by the sentence.
2. Keep analysis_steps short, concrete, and numbered.
3. Explain the overall clinical phenomenon first, then the likely mechanism.
4. Use at most 3 representative pairs; do not enumerate all query pairs.
5. Do not mention KG evidence, labels, or confidence.
"""

FEW_SHOT_ASSISTANT_EXPLANATION = """[ 
    { 
        "query": "tetracycline -> bismuth subsalicylate", 
        "analysis_steps": "1. Target Focus: Analyzing 'tetracycline -> bismuth subsalicylate'.\n2. Relevance Check: Sentence describes impaired absorption between these exact two drugs.\n3. KG Evidence Check: Shared node is ALB(Carrier), a non-specific transport protein.\n4. Synthesis: Bismuth subsalicylate physically interacts with tetracycline in the gut, reducing its systemic absorption.", 
        "mechanism_summary": "Physicochemical interaction: Bismuth subsalicylate likely chelates tetracycline in the gastrointestinal tract, reducing its absorption.", 
        "confidence_assessment": "Medium"
    } 
]
"""

FEW_SHOT_ASSISTANT_EXPLANATION_NO_KG = """[ 
    { 
        "query": "GLOBAL_PHENOMENON", 
        "analysis_steps": "1. Clinical phenomenon: the sentence reports reduced absorption during co-administration.\n2. Mechanistic interpretation: this is a pharmacokinetic interaction affecting drug exposure.\n3. Evidence compression: summarize representative pairs instead of enumerating all possible pair combinations.", 
        "mechanism_summary": "The clinical phenomenon is a mechanism-level DDI characterized by reduced absorption in co-administration.",
        "representative_pairs": ["tetracycline -> bismuth subsalicylate"]
    } 
]
"""


def _truncate_text(text: str, max_chars: int) -> str:
    s = str(text or "")
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip() + " ...[TRUNCATED]"
    return s


@lru_cache(maxsize=1)
def _fixed_prompt_roles() -> tuple:
    return (
        ("system", SYSTEM_PROMPT_WITH_KG),
        ("user", FEW_SHOT_USER),
        ("assistant", FEW_SHOT_ASSISTANT),
    )


@lru_cache(maxsize=1)
def _fixed_prompt_roles_no_kg() -> tuple:
    return (
        ("system", SYSTEM_PROMPT_NO_KG),
        ("user", FEW_SHOT_USER_NO_KG),
        ("assistant", FEW_SHOT_ASSISTANT_NO_KG),
    )


@lru_cache(maxsize=1)
def _fixed_prompt_roles_explanation() -> tuple:
    return (
        ("system", SYSTEM_PROMPT_EXPLANATION_WITH_KG),
        ("user", FEW_SHOT_USER),
        ("assistant", FEW_SHOT_ASSISTANT_EXPLANATION),
    )


@lru_cache(maxsize=1)
def _fixed_prompt_roles_explanation_no_kg() -> tuple:
    return (
        ("system", SYSTEM_PROMPT_EXPLANATION_NO_KG),
        ("user", FEW_SHOT_USER_NO_KG),
        ("assistant", FEW_SHOT_ASSISTANT_EXPLANATION_NO_KG),
    )


def build_messages_for_entry(
    entry: Dict[str, Any],
    max_kg_evidence_chars: int = 0,
    max_sentence_chars: int = 0,
    generation_mode: str = "label_conditioned",
    ignore_kg_evidence: bool = False,
) -> List[Dict[str, str]]:
    instruction = entry.get("instruction", "Analyze the biological mechanisms based on KG evidence.")
    input_data = entry.get("input", {})

    sentence = _truncate_text(input_data.get("sentence", ""), max_sentence_chars)
    raw_kg_evidence = "" if ignore_kg_evidence else str(input_data.get("kg_evidence", "") or "").strip()
    has_kg_evidence = bool(raw_kg_evidence)
    kg_evidence = _truncate_text(raw_kg_evidence, max_kg_evidence_chars)
    query_group = get_query_group_text(input_data)
    gold_label = input_data.get("gold_label", "Mechanism")
    if generation_mode not in {"label_conditioned", "explanation_only", "distill_reasoning_without_kg"}:
        raise ValueError("generation_mode must be one of: label_conditioned, explanation_only, distill_reasoning_without_kg")

    if generation_mode == "label_conditioned":
        if has_kg_evidence:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}",
  "kg_evidence": "{kg_evidence}",
  "target_expert_label": "{gold_label}"
}}"""
            grounding_line = "Please use sentence text and kg_evidence only."
        else:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}",
  "target_expert_label": "{gold_label}"
}}"""
            grounding_line = "No external KG evidence is provided for this sample. Please use sentence text only."

        current_user = f"""Instruction: {instruction}

Input Data:
{input_block}

CRITICAL REMINDER: You must ONLY analyze the exact drug pair(s) listed in the "query_group" field above. Do not analyze any other drugs. 
{grounding_line}
Please generate the detailed analysis steps that logically lead to the given 'target_expert_label' ({gold_label}).
Output the exact JSON format, ensuring the final "label" strictly matches "{gold_label}".
Output:"""

        roles = _fixed_prompt_roles() if has_kg_evidence else _fixed_prompt_roles_no_kg()
    else:
        force_no_kg = ignore_kg_evidence or generation_mode == "distill_reasoning_without_kg"
        if force_no_kg:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}"
}}"""
            grounding_line = "Use sentence text only. Do not reference KG evidence, labels, or confidence."

            current_user = f"""Instruction: {instruction}

Input Data:
{input_block}

CRITICAL REMINDER: Focus on the overall clinical phenomenon described by the sentence.
Use the query_group field only as evidence candidates and choose at most 3 representative pairs.
{grounding_line}
Please generate a concise explanation-only answer optimized for evaluation.
Return JSON list with fields: query, analysis_steps, mechanism_summary, representative_pairs.
Set query to "GLOBAL_PHENOMENON".
analysis_steps should contain 2-4 short numbered steps.
Output:"""

            roles = _fixed_prompt_roles_explanation_no_kg()
        elif has_kg_evidence:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}",
  "kg_evidence": "{kg_evidence}"
}}"""
            grounding_line = "Please use sentence text and kg_evidence only."

            current_user = f"""Instruction: {instruction}

Input Data:
{input_block}

CRITICAL REMINDER: You must ONLY analyze the exact drug pair(s) listed in the "query_group" field above. Do not analyze any other drugs. 
{grounding_line}
Please generate a structured explanation only.
Return JSON list with fields: query, analysis_steps, mechanism_summary, confidence_assessment.
Do not output label fields.
Output:"""

            roles = _fixed_prompt_roles_explanation()
        else:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}"
}}"""
            grounding_line = "No external KG evidence is provided for this sample. Please use sentence text only."

            current_user = f"""Instruction: {instruction}

Input Data:
{input_block}

CRITICAL REMINDER: Focus on the overall clinical phenomenon described by the sentence.
Use the query_group field only as evidence candidates and choose at most 3 representative pairs.
{grounding_line}
Please generate a concise explanation-only answer optimized for evaluation.
Return JSON list with fields: query, analysis_steps, mechanism_summary, representative_pairs.
Set query to "GLOBAL_PHENOMENON".
analysis_steps should contain 2-4 short numbered steps.
Do not output label fields or confidence fields.
Output:"""

            roles = _fixed_prompt_roles_explanation_no_kg()
    messages = [{"role": role, "content": content} for role, content in roles]
    messages.append({"role": "user", "content": current_user})
    return messages


def load_finetune_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_finetune_data(data: List[Dict[str, Any]], json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _merge_existing_outputs(
    processed_data: List[Dict[str, Any]],
    output_json: str,
    overwrite: bool,
) -> int:
    if overwrite:
        return 0

    out_path = Path(output_json)
    if not out_path.exists():
        return 0

    try:
        existing = load_finetune_data(str(out_path))
    except Exception as e:
        print(f"警告: 读取已有输出失败，将全量重跑: {e}")
        return 0

    reused = 0
    max_idx = min(len(processed_data), len(existing))
    for i in range(max_idx):
        out_text = str(existing[i].get("output", "")).strip()
        if out_text:
            processed_data[i]["output"] = out_text
            reused += 1
    return reused


def call_llm_chat(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    retry_delay: float,
    max_output_tokens: int,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM响应缺少choices: {data}")

            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            if not content:
                content = (msg.get("reasoning_content") or "").strip()
            if not content:
                raise RuntimeError(f"LLM响应无content: {data}")
            return content

        except error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            last_err = RuntimeError(f"HTTP {e.code}: {e.reason} {body_text}".strip())
            if attempt < max_retries:
                sleep_s = retry_delay * attempt + random.uniform(0, 0.4)
                print(f"请求失败(第{attempt}/{max_retries})，{sleep_s:.1f}s后重试: {last_err}")
                time.sleep(sleep_s)
            else:
                break
        except (error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = retry_delay * attempt + random.uniform(0, 0.4)
                print(f"请求失败(第{attempt}/{max_retries})，{sleep_s:.1f}s后重试: {e}")
                time.sleep(sleep_s)
            else:
                break

    raise RuntimeError(f"LLM调用失败: {last_err}")


def generate_outputs_with_distillation(
    input_json: str,
    output_json: str,
    api_key: str,
    model: str = "qwen-plus",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    limit: int = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_workers: int = 1,
    save_every: int = DEFAULT_SAVE_EVERY,
    log_every: int = DEFAULT_LOG_EVERY,
    overwrite: bool = False,
    max_kg_evidence_chars: int = 0,
    max_sentence_chars: int = 0,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    request_retries: int = DEFAULT_REQUEST_RETRIES,
    request_retry_delay: float = DEFAULT_REQUEST_RETRY_DELAY,
    generation_mode: str = "label_conditioned",
    ignore_kg_evidence: bool = False,
):
    print(f"加载数据: {input_json}")
    raw_data = load_finetune_data(input_json)
    if limit:
        raw_data = raw_data[:limit]

    processed_data: List[Dict[str, Any]] = []
    for entry in raw_data:
        if not get_query_group_text(entry.get("input", {})):
            continue
        cloned = dict(entry)
        if overwrite:
            cloned["output"] = ""
        else:
            cloned.setdefault("output", "")
        processed_data.append(cloned)

    reused = _merge_existing_outputs(processed_data, output_json, overwrite)
    if reused:
        print(f"已复用已有输出: {reused} 条")

    pending_indices = [i for i, x in enumerate(processed_data) if not str(x.get("output", "")).strip()]
    print(f"总样本: {len(processed_data)}，待生成: {len(pending_indices)}")

    completed_now = 0
    start_time = time.time()
    failed_indices: List[int] = []

    workers = max(1, int(max_workers))
    print(f"并发线程: {workers}")

    def _run_one(data_idx: int):
        entry = processed_data[data_idx]
        messages = build_messages_for_entry(
            entry,
            max_kg_evidence_chars=max_kg_evidence_chars,
            max_sentence_chars=max_sentence_chars,
            generation_mode=generation_mode,
            ignore_kg_evidence=ignore_kg_evidence,
        )
        out_text = call_llm_chat(
            messages=messages,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=request_retries,
            retry_delay=request_retry_delay,
            max_output_tokens=max_new_tokens,
        )
        return data_idx, out_text

    if workers == 1:
        for idx, data_idx in enumerate(pending_indices, 1):
            try:
                _, out_text = _run_one(data_idx)
            except Exception as e:
                out_text = ""
                print(f"错误: 第 {idx} 条调用失败: {e}")
                failed_indices.append(data_idx)

            processed_data[data_idx]["output"] = out_text
            completed_now += 1

            if log_every > 0 and (idx % log_every == 0 or idx == len(pending_indices)):
                elapsed = time.time() - start_time
                print(f"进度: {completed_now}/{len(pending_indices)} | elapsed={elapsed:.1f}s")

            if save_every > 0 and completed_now % save_every == 0:
                checkpoint_data = []
                for rec in processed_data:
                    cleaned = dict(rec)
                    input_data = dict(cleaned.get("input", {}))
                    input_data.pop("gold_label", None)
                    if ignore_kg_evidence:
                        input_data.pop("kg_evidence", None)
                    cleaned["input"] = input_data
                    checkpoint_data.append(cleaned)
                save_finetune_data(checkpoint_data, output_json)
                print(f"checkpoint: 已完成 {completed_now}/{len(pending_indices)}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, data_idx): data_idx for data_idx in pending_indices}
            total = len(pending_indices)
            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                data_idx = futures[future]
                try:
                    _, out_text = future.result()
                except Exception as e:
                    out_text = ""
                    print(f"错误: data_idx={data_idx} 调用失败: {e}")
                    failed_indices.append(data_idx)

                processed_data[data_idx]["output"] = out_text
                completed_now += 1

                if log_every > 0 and (idx % log_every == 0 or idx == total):
                    elapsed = time.time() - start_time
                    print(f"进度: {completed_now}/{total} | elapsed={elapsed:.1f}s")

                if save_every > 0 and completed_now % save_every == 0:
                    checkpoint_data = []
                    for rec in processed_data:
                        cleaned = dict(rec)
                        input_data = dict(cleaned.get("input", {}))
                        input_data.pop("gold_label", None)
                        if ignore_kg_evidence:
                            input_data.pop("kg_evidence", None)
                        cleaned["input"] = input_data
                        checkpoint_data.append(cleaned)
                    save_finetune_data(checkpoint_data, output_json)
                    print(f"checkpoint: 已完成 {completed_now}/{total}")

    # 对失败项做一次顺序回填，避免并发中的单条异常导致最终结果留空
    if failed_indices:
        unique_failed = list(dict.fromkeys(failed_indices))
        print(f"发现 {len(unique_failed)} 条失败项，开始顺序回填重试...")
        for retry_idx, data_idx in enumerate(unique_failed, 1):
            try:
                _, out_text = _run_one(data_idx)
                if str(out_text).strip():
                    processed_data[data_idx]["output"] = out_text
                    print(f"回填成功: data_idx={data_idx}")
                else:
                    print(f"回填仍为空: data_idx={data_idx}")
            except Exception as e:
                print(f"回填失败: data_idx={data_idx}, 错误: {e}")

    final_data = []
    for rec in processed_data:
        cleaned = dict(rec)
        input_data = dict(cleaned.get("input", {}))
        input_data.pop("gold_label", None)
        if ignore_kg_evidence:
            input_data.pop("kg_evidence", None)
        cleaned["input"] = input_data
        final_data.append(cleaned)

    save_finetune_data(final_data, output_json)
    print(f"\n生成完成！已保存到: {output_json}")


def main():
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Generate DDI outputs for distillation")
    parser.add_argument("--config", default=None, help="实验配置文件路径（YAML/JSON），例如 configs/experiments/restart_explanation_bootstrap.yaml")
    parser.add_argument("--config_section", default="distillation_generation", help="配置文件中的蒸馏生成段名")
    parser.add_argument("--input", default=str(project_root / "data" / "finetune_dataset_input.json"))
    parser.add_argument("--output", default=str(project_root / "data" / "finetune_dataset.json"))
    parser.add_argument("--api_key", default=(os.environ.get("LLM_DISTILLATION_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")))
    parser.add_argument("--base_url", default=(os.environ.get("LLM_DISTILLATION_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="单条最大生成token")
    parser.add_argument("--max_workers", type=int, default=4, help="并发线程数（API请求并发）")
    parser.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY, help="每N条保存checkpoint")
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY, help="每N条打印一次进度")
    parser.add_argument("--overwrite", action="store_true", help="忽略已有输出并全量重跑")
    parser.add_argument(
        "--generation_mode",
        default="label_conditioned",
        choices=["label_conditioned", "explanation_only", "distill_reasoning_without_kg"],
        help="label_conditioned=生成推理+标签；explanation_only=解释输出；distill_reasoning_without_kg=强制无KG蒸馏推理",
    )
    parser.add_argument(
        "--ignore_kg_evidence",
        action="store_true",
        help="忽略输入中的kg_evidence字段，强制使用sentence-only提示并在输出中删除kg_evidence",
    )
    args = parser.parse_args()

    section: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
        raw_section = cfg.get(args.config_section, {}) if isinstance(cfg, dict) else {}
        if isinstance(raw_section, dict):
            section = raw_section

    input_value = str(section.get("input", args.input))
    output_value = str(section.get("output", args.output))
    api_key_value = section.get("api_key", args.api_key)
    base_url_value = str(section.get("base_url", args.base_url))
    model_value = str(section.get("model", args.model))
    limit_value = section.get("limit", args.limit)
    max_new_tokens_value = int(section.get("max_new_tokens", args.max_new_tokens))
    max_workers_value = int(section.get("max_workers", args.max_workers))
    save_every_value = int(section.get("save_every", args.save_every))
    log_every_value = int(section.get("log_every", args.log_every))
    generation_mode_value = str(section.get("generation_mode", args.generation_mode))
    overwrite_value = bool(section.get("overwrite", args.overwrite))
    ignore_kg_value = bool(section.get("ignore_kg_evidence", args.ignore_kg_evidence))

    if not api_key_value:
        raise ValueError("缺少蒸馏API Key。请设置 LLM_DISTILLATION_API_KEY（或 DASHSCOPE_API_KEY）或使用 --api_key。")

    model_name = model_value.strip() or "qwen-plus"

    input_path = Path(input_value)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    max_new_tokens = int(max_new_tokens_value)
    max_workers = int(max_workers_value)
    save_every = int(save_every_value)
    log_every = int(log_every_value)

    print("=" * 60)
    print("使用蒸馏 LLM 生成输出")
    print(
        f"simple mode | model={model_name} | max_new_tokens={max_new_tokens} | "
        f"max_workers={max_workers} | save_every={save_every} | log_every={log_every} | ignore_kg_evidence={ignore_kg_value}"
    )
    print("=" * 60)

    generate_outputs_with_distillation(
        input_json=str(input_path),
        output_json=str(Path(output_value)),
        api_key=api_key_value,
        model=model_name,
        base_url=base_url_value,
        limit=limit_value,
        max_new_tokens=max_new_tokens,
        max_workers=max_workers,
        save_every=save_every,
        log_every=log_every,
        overwrite=overwrite_value,
        generation_mode=generation_mode_value,
        ignore_kg_evidence=ignore_kg_value,
    )


if __name__ == "__main__":
    main()
