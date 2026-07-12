import json
import time
from typing import Any, Dict, List
from urllib import error, request

from pipelines.query_utils import get_query_group_text
from .judge_prompts import SYSTEM_PROMPT, build_user_prompt
from .result_normalizer import normalize_judge_result


class QwenMaxJudge:
    def __init__(
        self,
        api_key: str = "",
        model_id: str = "qwen-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 4,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ):
        if not api_key:
            raise ValueError("QwenMaxJudge requires a non-empty api_key")
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        endpoint = self.base_url + "/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"Invalid response: {data}")
                content = (choices[0].get("message") or {}).get("content", "")
                content = str(content).strip()
                if not content:
                    raise RuntimeError(f"Empty content: {data}")
                return content
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    if self.verbose:
                        print(f"[Judge] request failed (attempt {attempt}/{self.max_retries}), retrying: {exc}")
                    time.sleep(self.retry_delay * attempt)
                    continue
        raise RuntimeError(f"qwen-max judge call failed: {last_err}")

    def judge_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sentence = str(sample.get("sentence", ""))
        query_group = get_query_group_text(sample)
        explanation = str(sample.get("reasoning", ""))
        kg_evidence = str(sample.get("kg_evidence", ""))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    sentence=sentence,
                    query_group=query_group,
                    explanation=explanation,
                    kg_evidence=kg_evidence,
                ),
            },
        ]

        raw = self._chat(messages)
        return normalize_judge_result(raw_text=raw, payload=raw)

    def batch_judge(self, samples: List[Dict[str, Any]], checkpoint_every: int = 50) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self.verbose:
            print(f"[Judge] start: model={self.model_id}, samples={len(samples)}")
        for idx, sample in enumerate(samples, 1):
            try:
                result = self.judge_sample(sample)
            except error.HTTPError as exc:
                result = normalize_judge_result(
                    payload={"short_rationale": f"HTTPError: {exc}", "mechanism_gaps": []}
                )
            except Exception as exc:
                result = normalize_judge_result(
                    payload={"short_rationale": f"Judge error: {exc}", "mechanism_gaps": []}
                )
            results.append(result)
            if idx % checkpoint_every == 0:
                if self.verbose:
                    print(f"[Judge] processed {idx}/{len(samples)}")
        if self.verbose:
            print(f"[Judge] done: processed {len(samples)} samples")
        return results
