"""
使用 MedGemma 模型为微调数据集补充输出（改进版）
读取已构造的输入数据（包含 sentence / entity_profiles / queries_with_shared），
按提示工程规则构建 System + Few-Shot + 当前样本消息，调用 MedGemma 生成"Mechanism Analysis"。
"""

import json
import torch
from typing import List, Dict, Any
from pathlib import Path
import argparse
import os
import time
from functools import lru_cache
from modelscope import AutoTokenizer, AutoModelForCausalLM
from pipelines.query_utils import get_query_group_text

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_SAVE_EVERY = 100
DEFAULT_LOG_EVERY = 10

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
Generate explanation-only outputs from sentence text (no external KG evidence).

### OUTPUT FORMAT
Output a JSON list of objects:
[{"query": "A -> B", "analysis_steps": "Step 1... 2...", "mechanism_summary": "<brief>", "confidence_assessment": "High/Medium/Low"}]
Do not output any label field.
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
        "query": "tetracycline -> bismuth subsalicylate", 
        "analysis_steps": "1. Target Focus: Analyzing 'tetracycline -> bismuth subsalicylate'.\n2. Relevance Check: Sentence explicitly states absorption is impaired between these exact two drugs.\n3. Text-grounded Interpretation: The interaction is an absorption change, indicating a pharmacokinetic mechanism.", 
        "mechanism_summary": "The sentence indicates reduced tetracycline absorption when co-administered with bismuth subsalicylate.", 
        "confidence_assessment": "High"
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
    """缓存固定 few-shot 提示词，避免每条样本重复组装。"""
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
) -> List[Dict[str, str]]:
    """
    构建消息列表用于模型生成（Conditioned CoT Generation）
    从 input.gold_label 读取人工标注标签，引导模型生成与标签一致的推理链。
    """
    instruction = entry.get("instruction", "Analyze the biological mechanisms based on KG evidence.")
    input_data = entry.get("input", {})

    sentence = _truncate_text(input_data.get("sentence", ""), max_sentence_chars)
    raw_kg_evidence = str(input_data.get("kg_evidence", "") or "").strip()
    has_kg_evidence = bool(raw_kg_evidence)
    kg_evidence = _truncate_text(raw_kg_evidence, max_kg_evidence_chars)
    query_group = get_query_group_text(input_data)
    # 从原始数据中读取人工标注的真实标签
    gold_label = input_data.get("gold_label", "Mechanism")

    if generation_mode not in {"label_conditioned", "explanation_only"}:
        raise ValueError("generation_mode must be one of: label_conditioned, explanation_only")

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
        if has_kg_evidence:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}",
  "kg_evidence": "{kg_evidence}"
}}"""
            grounding_line = "Please use sentence text and kg_evidence only."
        else:
            input_block = f"""{{
  "sentence": "{sentence}",
  "query_group": "{query_group}"
}}"""
            grounding_line = "No external KG evidence is provided for this sample. Please use sentence text only."

        current_user = f"""Instruction: {instruction}

Input Data:
{input_block}

CRITICAL REMINDER: You must ONLY analyze the exact drug pair(s) listed in the "query_group" field above. Do not analyze any other drugs.
{grounding_line}
Please generate a structured explanation only.
Return JSON list with fields: query, analysis_steps, mechanism_summary, confidence_assessment.
Do not output label fields.
Output:"""
        roles = _fixed_prompt_roles_explanation() if has_kg_evidence else _fixed_prompt_roles_explanation_no_kg()
    messages = [{"role": role, "content": content} for role, content in roles]
    messages.append({"role": "user", "content": current_user})

    return messages


# ============================
# 4. 数据加载与保存
# ============================

def load_finetune_data(json_path: str) -> List[Dict[str, Any]]:
    """加载微调数据集"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_finetune_data(data: List[Dict[str, Any]], json_path: str):
    """保存微调数据集"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _generate_single_from_prompt_text(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    """基于完整 prompt 文本进行单条生成（回退路径）。"""
    single_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).to(model.device)
    single_input_len = single_inputs["input_ids"].shape[-1]
    generation = model.generate(
        **single_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        generation[0][single_input_len:],
        skip_special_tokens=True,
    ).strip()


def _generate_batch_from_prompt_texts(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs: List[str] = []
    for row in range(len(prompt_texts)):
        input_len = int(input_lens[row])
        out_text = tokenizer.decode(
            generation[row][input_len:],
            skip_special_tokens=True,
        ).strip()
        outputs.append(out_text)
    return outputs


def _merge_existing_outputs(
    processed_data: List[Dict[str, Any]],
    output_json: str,
    overwrite: bool,
) -> int:
    """从已有输出文件恢复可复用的 output，用于断点续跑。"""
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


# ============================
# 5. 生成逻辑 (Pipeline)
# ============================

def generate_outputs_with_model(
    input_json: str,
    output_json: str,
    model_path: str = "models/google/medgemma-27b-text-it",
    limit: int = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 1,
    save_every: int = DEFAULT_SAVE_EVERY,
    log_every: int = DEFAULT_LOG_EVERY,
    overwrite: bool = False,
    max_kg_evidence_chars: int = 0,
    max_sentence_chars: int = 0,
    generation_mode: str = "label_conditioned",
):
    """使用 MedGemma 模型逐条生成输出。"""
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

    print(f"加载模型: {model_path}")
    if torch.cuda.is_available():
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    project_root = Path(__file__).parent.parent
    mp = Path(model_path)
    if not mp.is_absolute():
        mp = project_root / mp

    device_map = "balanced" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "auto"
    use_local = False
    if mp.exists() and mp.is_dir():
        if (mp / "config.json").exists() or (mp / "tokenizer_config.json").exists():
            use_local = True

    remote_id = "google/medgemma-27b-text-it"
    if use_local:
        try:
            print(f"加载本地模型: {mp}")
            model = AutoModelForCausalLM.from_pretrained(str(mp), dtype=torch.bfloat16, device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(str(mp))
        except Exception as e:
            print(f"本地模型加载失败，回退到远端ID: {remote_id}，错误: {e}")
            model = AutoModelForCausalLM.from_pretrained(remote_id, dtype=torch.bfloat16, device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(remote_id)
    else:
        print(f"未检测到完整本地模型文件，使用远端ID: {remote_id}")
        model = AutoModelForCausalLM.from_pretrained(remote_id, dtype=torch.bfloat16, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(remote_id)

    if hasattr(model, "config"):
        try:
            model.config.use_cache = True
        except Exception:
            pass

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        try:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
            if getattr(model.config, "eos_token_id", None) is None:
                model.config.eos_token_id = tokenizer.eos_token_id
        except Exception:
            pass

    total = len(processed_data)
    pending_indices = [i for i, x in enumerate(processed_data) if not str(x.get("output", "")).strip()]
    print(f"总样本: {total}，待生成: {len(pending_indices)}")

    prompt_text_cache: Dict[int, str] = {}
    completed_now = 0
    start_time = time.time()

    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for offset in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[offset: offset + batch_size]
            prompt_texts: List[str] = []
            for data_idx in batch_indices:
                prompt_text = prompt_text_cache.get(data_idx)
                if prompt_text is None:
                    messages = build_messages_for_entry(
                        processed_data[data_idx],
                        max_kg_evidence_chars=max_kg_evidence_chars,
                        max_sentence_chars=max_sentence_chars,
                        generation_mode=generation_mode,
                    )
                    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    prompt_text_cache[data_idx] = prompt_text
                prompt_texts.append(prompt_text)

            outputs = [""] * len(batch_indices)
            try:
                outputs = _generate_batch_from_prompt_texts(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompt_texts,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                print(f"警告: batch 生成失败，回退单条: {e}")
                for row, prompt_text in enumerate(prompt_texts):
                    try:
                        outputs[row] = _generate_single_from_prompt_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_text=prompt_text,
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception as single_e:
                        outputs[row] = ""
                        print(f"错误: 第 {completed_now + row + 1} 条数据生成失败: {single_e}")

            for row, data_idx in enumerate(batch_indices):
                processed_data[data_idx]["output"] = outputs[row]
                completed_now += 1

                if log_every > 0 and (completed_now % log_every == 0 or completed_now == len(pending_indices)):
                    elapsed = time.time() - start_time
                    print(f"进度: {completed_now}/{len(pending_indices)} | elapsed={elapsed:.1f}s")

                if save_every > 0 and completed_now % save_every == 0:
                    checkpoint_data = []
                    for entry in processed_data:
                        cleaned = dict(entry)
                        input_data = dict(cleaned.get("input", {}))
                        input_data.pop("gold_label", None)
                        cleaned["input"] = input_data
                        checkpoint_data.append(cleaned)
                    save_finetune_data(checkpoint_data, output_json)
                    print(f"checkpoint: 已完成 {completed_now}/{len(pending_indices)}")

                if torch.cuda.is_available() and completed_now % max(20, save_every) == 0:
                    torch.cuda.empty_cache()

    final_data = []
    for entry in processed_data:
        cleaned = dict(entry)
        input_data = dict(cleaned.get("input", {}))
        input_data.pop("gold_label", None)
        cleaned["input"] = input_data
        final_data.append(cleaned)

    save_finetune_data(final_data, output_json)
    print(f"\n生成完成！已保存到: {output_json}")


# ============================
# 6. 主函数
# ============================

def main():
    """主函数：生成输出"""
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Generate DDI outputs with MedGemma")
    parser.add_argument("--input", default=str(project_root / "data" / "finetune_dataset_input.json"))
    parser.add_argument("--output", default=str(project_root / "data" / "finetune_dataset.json"))
    parser.add_argument("--model_path", default=str(project_root / "models" / "google" / "medgemma-27b-text-it"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="单条最大生成token")
    parser.add_argument("--batch_size", type=int, default=1, help="并行批大小（GPU批量并发）")
    parser.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY, help="每N条保存checkpoint")
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY, help="每N条打印一次进度")
    parser.add_argument("--overwrite", action="store_true", help="忽略已有输出并全量重跑")
    parser.add_argument(
        "--generation_mode",
        default="label_conditioned",
        choices=["label_conditioned", "explanation_only"],
        help="label_conditioned=生成推理+标签；explanation_only=只生成解释不输出标签",
    )
    args = parser.parse_args()

    max_new_tokens = int(args.max_new_tokens)
    batch_size = int(args.batch_size)
    save_every = int(args.save_every)
    log_every = int(args.log_every)

    print(f"simple mode | max_new_tokens={max_new_tokens} | batch_size={batch_size} | save_every={save_every} | log_every={log_every}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("生成 MedGemma 输出")
    print("=" * 60)
    
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        print("请先运行 build_finetune_dataset.py 生成输入数据")
        return

    # 生成输出（仅处理前10条作为示例）
    generate_outputs_with_model(
        input_json=str(input_path),
        output_json=str(output_path),
        model_path=str(args.model_path),
        limit=args.limit,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        save_every=save_every,
        log_every=log_every,
        overwrite=args.overwrite,
        generation_mode=args.generation_mode,
    )


if __name__ == "__main__":
    main()
