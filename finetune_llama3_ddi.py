"""
Finetune Meta-Llama-3-8B-Instruct on DDI finetune_dataset.json using LoRA (QLoRA-ready).
- Input: data/finetune_dataset.json (fields: instruction, input.sentence, input.query_group, input.kg_evidence, output)
- Output: results/llama3_ddi_lora with adapter weights.
"""

import argparse
import json
import inspect
from pathlib import Path
from typing import Dict, Any, List

import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from pipelines.query_utils import get_query_group_text


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
Given clinical text and KG evidence, provide a detailed mechanism analysis and a final DDI label.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output must be a JSON list; each item should include:
query, analysis_steps, mechanism_summary, confidence_assessment, label.
""",
        "output_hint": "Return detailed reasoning chain and final label in JSON list format.",
    },
    "explanation_with_kg": {
        "include_kg": True,
        "system_prompt": """You are an expert Clinical Pharmacologist and Graph Reasoning Assistant.
Given clinical text and KG evidence, provide detailed mechanism explanations only.

Output must be a JSON list; each item should include:
query, analysis_steps, mechanism_summary.
Do not output DDI labels.
""",
        "output_hint": "Return detailed reasoning chain only (no label field, no confidence field).",
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
Given sentence text only (no external KG evidence), provide detailed reasoning and DDI labels.

Valid labels are exactly:
- Mechanism
- Effect
- Advice
- Int
- False

Output must be a JSON list; each item should include:
query, analysis_steps, mechanism_summary, confidence_assessment, label.
""",
        "output_hint": "No external KG is available. Use sentence text only, with detailed reasoning.",
    },
    "explanation_without_kg": {
        "include_kg": False,
        "system_prompt": """You are an expert Clinical Pharmacologist.
Given sentence text only (no external KG evidence), provide concise, evaluation-friendly phenomenon-level explanations only.

Output must be a JSON list; each item should include:
query, analysis_steps, mechanism_summary, representative_pairs.
Do not output DDI labels.
Keep analysis_steps short, numbered, and grounded in the sentence.
Set query to "GLOBAL_PHENOMENON" and keep representative_pairs to at most 3.
""",
        "output_hint": "No external KG is available. Return phenomenon-level reasoning only (no label field, no confidence field, max 3 representative pairs).",
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
    "output_hint": "No external KG is available. Output one final label only (Mechanism/Effect/Advice/Int/False).",
    },
}


def resolve_prompt_mode(requested_mode: str, data_path: Path, sample: Dict[str, Any]) -> str:
    """Resolve requested prompt mode and support backward-compatible aliases."""
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
        mode = _infer_prompt_mode(data_path=data_path, sample=sample)
    if mode not in PROMPT_PROFILES:
        raise ValueError(f"Unsupported prompt mode: {requested_mode}")
    return mode


def _infer_prompt_mode(data_path: Path, sample: Dict[str, Any]) -> str:
    """Infer prompt mode from file name and sample schema when --prompt_mode=auto."""
    name = data_path.name.lower()
    if "label_only_without_kg" in name:
        return "label_only_without_kg"
    if "explanation_without_kg" in name:
        return "explanation_without_kg"
    if "reasoning_without_kg" in name:
        return "reasoning_without_kg"
    if "explanation_with_kg" in name:
        return "explanation_with_kg"
    if "label_only_with_kg" in name:
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


def build_prompt(example: Dict[str, Any], prompt_mode: str) -> List[Dict[str, str]]:
    """Construct chat-formatted SFT data aligned with selected ablation prompt mode."""
    instruction = example.get("instruction", "Analyze the biological mechanisms based on KG evidence.")
    inp = example.get("input", {})
    sentence = str(inp.get("sentence", "") or "")
    kg_evidence = str(inp.get("kg_evidence", "") or "")
    query_group = get_query_group_text(inp)
    output = example.get("output", "")

    profile = PROMPT_PROFILES[prompt_mode]
    include_kg = bool(profile["include_kg"])

    input_payload = {
        "sentence": sentence,
        "query_group": query_group,
    }
    if include_kg:
        input_payload["kg_evidence"] = kg_evidence

    input_json = json.dumps(input_payload, ensure_ascii=False, indent=2)

    user_msg = f"""Instruction: {instruction}

Input Data:
{input_json}

Prompt Strategy: {prompt_mode}
{profile["output_hint"]}

Output:"""

    messages = [
        {"role": "system", "content": profile["system_prompt"]},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    return messages


def tokenize_dataset(dataset, tokenizer, prompt_mode: str):
    def _convert(example):
        messages = build_prompt(example, prompt_mode=prompt_mode)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return dataset.map(_convert, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="Finetune Meta-Llama-3-8B-Instruct on DDI dataset with LoRA")
    parser.add_argument("--model_id", type=str, default="models/Meta-Llama-3-8B-Instruct", help="Base model path or hub id")
    parser.add_argument("--data_path", type=str, default="data/finetune_dataset.json", help="Path to finetune dataset json")
    parser.add_argument("--output_dir", type=str, default="results/llama3_ddi_lora", help="Where to save adapter")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--train_test_split", type=float, default=0.05, help="Fraction for eval")
    parser.add_argument("--multi_gpu_shard", action="store_true", help="Shard base model across multiple GPUs in one process")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for SFTTrainer")
    parser.add_argument("--bf16", action="store_true", help="Force bfloat16; default auto if CUDA")
    parser.add_argument("--allow_single_process_multi_gpu", action="store_true", help="Allow Trainer DataParallel in single process (not recommended on Windows)")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA loading (default is enabled)")
    parser.add_argument("--max_memory_ratio", type=float, default=0.82, help="Per-GPU memory ratio used by model sharding")
    parser.add_argument("--offload_folder", type=str, default="offload", help="Folder for offload buffers when sharding")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="auto",
        choices=sorted(PROMPT_MODES),
        help="Prompt engineering mode: auto/full/explanation_with_kg/explanation_without_kg/label_only_with_kg/reasoning_without_kg/label_only_without_kg",
    )
    args = parser.parse_args()

    # Reduce fragmentation per PyTorch docs if user hits OOM often (new env var name)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # On Windows single-process multi-GPU often falls back to DataParallel and can OOM on GPU0.
    # Default to single GPU unless explicitly allowed or torchrun (LOCAL_RANK) is used.
    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        is_torchrun = "LOCAL_RANK" in os.environ
        if visible_count > 1 and (not args.allow_single_process_multi_gpu) and (not is_torchrun) and (not args.multi_gpu_shard):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("Detected multi-GPU in single-process mode; forcing single GPU (cuda:0) to avoid DataParallel OOM/NCCL issues.")

    # Use standard DDP: don't shard the model with device_map; let Trainer handle multi-GPU via torchrun
    torch_dtype = torch.bfloat16 if (args.bf16 or torch.cuda.is_available()) else torch.float32
    use_4bit = not args.no_4bit

    print(f"Loading model from {args.model_id} ...")
    from_pretrained_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if use_4bit:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            from_pretrained_kwargs["quantization_config"] = quant_cfg
            print("Using 4-bit QLoRA loading (nf4 + double quant).")
        except Exception as e:
            use_4bit = False
            print(f"Warning: failed to enable 4-bit quantization, fallback to full precision load. Error: {e}")

    # Optional single-process multi-GPU sharding (model parallel) for Windows users avoiding torchrun
    if args.multi_gpu_shard and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        try:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory
                headroom = int(total * args.max_memory_ratio)
                # accelerate expects integer GPU indices or 'cpu'/'disk' keys
                max_memory[i] = headroom
            from_pretrained_kwargs["device_map"] = "balanced"
            from_pretrained_kwargs["max_memory"] = max_memory
            from_pretrained_kwargs["offload_folder"] = args.offload_folder
            pretty = {f"cuda:{k}": v for k, v in max_memory.items()}
            print(f"Using single-process multi-GPU sharding with device_map='balanced': {pretty}")
        except Exception as e:
            print(f"Warning: failed to configure multi_gpu_shard, falling back to single GPU. Error: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **from_pretrained_kwargs,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load data
    data_path = Path(args.data_path)
    dataset = load_dataset("json", data_files=str(data_path))
    dataset = dataset["train"]

    # Keep only samples with non-empty generated outputs.
    dataset = dataset.filter(lambda x: bool(str(x.get("output", "")).strip()))

    sample = dataset[0] if len(dataset) > 0 else {}
    resolved_prompt_mode = resolve_prompt_mode(args.prompt_mode, data_path=data_path, sample=sample)
    print(f"Prompt engineering mode: {resolved_prompt_mode}")

    dataset = dataset.train_test_split(test_size=args.train_test_split, seed=42)

    # Tokenize with chat template
    tokenized_train = tokenize_dataset(dataset["train"], tokenizer, prompt_mode=resolved_prompt_mode)
    tokenized_eval = tokenize_dataset(dataset["test"], tokenizer, prompt_mode=resolved_prompt_mode)

    # For memory efficiency on 8B with LoRA, enable gradient checkpointing and TF32
    try:
        # Use the new TF32 API to avoid deprecation warnings
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass

    # Disable cache during training for checkpointing
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass

    # Build TrainingArguments with version-compat keys
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    if "gradient_checkpointing" in ta_sig.parameters:
        ta_kwargs["gradient_checkpointing"] = True
    if "gradient_checkpointing_kwargs" in ta_sig.parameters:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "ddp_find_unused_parameters" in ta_sig.parameters:
        ta_kwargs["ddp_find_unused_parameters"] = False
    if "optim" in ta_sig.parameters:
        ta_kwargs["optim"] = "paged_adamw_8bit" if use_4bit else "adamw_torch"
    # Avoid setting deprecated TF32 TrainingArguments switch; we already set new torch backend API above.

    training_args = TrainingArguments(**ta_kwargs)

    # Build SFTTrainer kwargs based on its signature to avoid unexpected-arg errors
    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
    }
    if "tokenizer" in sft_sig.parameters:
        sft_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in sft_sig.parameters:
        sft_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_len
    if "packing" in sft_sig.parameters:
        sft_kwargs["packing"] = False

    trainer = SFTTrainer(**sft_kwargs)

    print("Starting training...")
    trainer.train()
    print("Training complete. Saving adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
