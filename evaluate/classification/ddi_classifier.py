import time
from typing import Any, Dict, List, Tuple

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

from .output_parser import extract_prediction
from .prompt_builder import create_prompt_shot, extract_drugs


class DDIClassifier:
    def __init__(self, model_id: str, ddi_types: List[str], max_new_tokens: int = 256):
        self.model_id = model_id
        self.ddi_types = ddi_types
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def predict_one(self, text: str) -> Tuple[str, str, str]:
        if self.model is None or self.tokenizer is None:
            self.load()

        drug1, drug2, context = extract_drugs(text)
        user_prompt = create_prompt_shot(drug1, drug2, context)
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert specializing in drug-drug interactions.",
            },
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        decoded = self.tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
        relation, reasoning = extract_prediction(decoded, self.ddi_types)
        return relation, reasoning, decoded

    def predict_dataset(self, samples: List[Tuple[str, str]]) -> Dict[str, Any]:
        predictions: List[str] = []
        reasoning_chains: List[str] = []
        raw_outputs: List[str] = []

        start = time.time()
        for i, (_, text) in enumerate(samples, 1):
            pred, reasoning, raw = self.predict_one(text)
            predictions.append(pred)
            reasoning_chains.append(reasoning)
            raw_outputs.append(raw)
            if i % 10 == 0:
                elapsed = time.time() - start
                print(f"Processed {i}/{len(samples)} in {elapsed:.2f}s")

        return {
            "predictions": predictions,
            "reasoning_chains": reasoning_chains,
            "raw_outputs": raw_outputs,
        }
