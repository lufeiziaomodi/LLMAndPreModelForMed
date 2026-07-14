"""Agent Loop: 反思-重试推理引擎（reasoning-with-critique loop）。

工程动机：无论是外部大模型直接推理，还是本地 LoRA 微调推理，都可能在第一次输出上
出现幻觉、机制断裂、方向反转等问题。本模块提供一个**样本级**的反思循环：

    generate() → critique() → 若 faithfulness/judge 低于阈值 → 把 gap 反馈拼进
    上下文再 generate() → 循环直到达标或达最大轮数。

该模块**与生成器解耦**：调用方只需实现 `ReasonerFn`：
    (messages: List[Dict], round_idx: int) -> str

因此：
- 外部 LLM 场景：把 `call_llm_chat(...)` 包一层就是 ReasonerFn；
- 本地 LoRA 场景：把 `model.generate(...) + tokenizer.decode(...)` 包一层就是 ReasonerFn。

评估器同样解耦，通过 `Critic` 抽象注入。默认给出：
- `FaithfulnessCritic`  纯本地计算，无需 API
- `JudgeCritic`         调 qwen-max，需 API key

阈值默认取"底线"值（faithfulness.strict_supported 且 hallucination_rate <= 阈值；
judge.mechanism_overall_score >= 阈值），可通过 AgentLoopConfig 调整。

产出：`AgentLoopTrace` — 每轮 messages / raw output / critique 全量落盘，
方便后续观察循环是否收敛。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

# 生成器：给定完整 messages 序列 + 当前轮次，返回模型原始文本输出。
# 由调用方（外部 LLM 或 本地 LoRA）实现具体调用逻辑。
ReasonerFn = Callable[[List[Dict[str, str]], int], str]


@dataclass
class CritiqueResult:
    """一次评估结果的统一表达。"""

    passed: bool
    score: float                     # 0.0~1.0（越高越好）
    reason: str                      # 简短原因，用于反馈进上下文
    detail: Dict[str, Any] = field(default_factory=dict)


class Critic:
    """评估器抽象。子类实现 `evaluate(sample, output)`。"""

    name: str = "critic"

    def evaluate(self, sample: Dict[str, Any], output: str) -> CritiqueResult:
        raise NotImplementedError


@dataclass
class AgentLoopConfig:
    """反思循环的行为配置。"""

    max_rounds: int = 3                  # 包括第 1 轮"初始生成"在内的总轮数
    stop_when_all_pass: bool = True      # 所有 critic 都 passed 就提前退出
    persist_full_trace: bool = True      # 是否落盘每轮完整 messages + output
    verbose: bool = False


@dataclass
class RoundRecord:
    """单轮记录。"""

    round_idx: int
    output: str
    critiques: List[Dict[str, Any]]      # 每个 critic 的 CritiqueResult.asdict
    feedback_injected: bool              # 该轮之前是否注入了上一轮 feedback


@dataclass
class AgentLoopTrace:
    """整条样本的循环轨迹。"""

    rounds: List[RoundRecord] = field(default_factory=list)
    final_output: str = ""
    final_passed: bool = False
    final_score: float = 0.0
    stopped_reason: str = ""             # "all_pass" | "max_rounds" | "generator_error"


# ---------------------------------------------------------------------------
# 内置 Critic 实现
# ---------------------------------------------------------------------------


class FaithfulnessCritic(Critic):
    """基于 evaluate.explanation.faithfulness 的本地事实度评估。

    通过 `passed = coverage_ratio >= min_coverage 且 hallucination_rate <= max_hall`
    判断是否达标。同时构造一段可读的 feedback，把 unsupported/partial claims 明确
    列进去，便于模型下一轮修正。
    """

    name = "faithfulness"

    def __init__(
        self,
        min_coverage: float = 0.4,
        max_hallucination: float = 0.5,
        max_feedback_claims: int = 5,
    ):
        self.min_coverage = float(min_coverage)
        self.max_hallucination = float(max_hallucination)
        self.max_feedback_claims = int(max_feedback_claims)

    def evaluate(self, sample: Dict[str, Any], output: str) -> CritiqueResult:
        # 延迟 import，避免循环依赖 + 让不装 sklearn 的环境也能 import 本模块
        from evaluate.explanation.faithfulness import compute_faithfulness_for_sample

        metric = compute_faithfulness_for_sample(
            sentence=str(sample.get("sentence", "")),
            explanation=output,
            predicted_label=str(sample.get("predicted_label", "") or sample.get("gold_label", "")),
            kg_evidence=str(sample.get("kg_evidence", "")),
            queries=str(sample.get("query_group", "") or sample.get("queries", "")),
        )
        coverage = float(metric.get("coverage_ratio", 0.0))
        hall = float(metric.get("hallucination_rate", 1.0))
        passed = (coverage >= self.min_coverage) and (hall <= self.max_hallucination)
        score = max(0.0, min(1.0, coverage - 0.5 * hall + 0.5))  # 单值组合，仅用于排序展示

        # 构造 feedback：把不通过的 claim 拼成 bullet
        unsupported = metric.get("unsupported_claims", []) or []
        partial = metric.get("partial_supported_claims", []) or []
        gap_bullets: List[str] = []
        for c in unsupported[: self.max_feedback_claims]:
            claim_text = str(c.get("claim", "")).strip()
            if claim_text:
                gap_bullets.append(f"- UNSUPPORTED by evidence: {claim_text}")
        remaining = self.max_feedback_claims - len(gap_bullets)
        for c in partial[:remaining]:
            claim_text = str(c.get("claim", "")).strip()
            if claim_text:
                gap_bullets.append(f"- PARTIALLY supported (needs tightening): {claim_text}")

        if not gap_bullets:
            reason = (
                f"faithfulness low: coverage={coverage:.2f} (min {self.min_coverage:.2f}), "
                f"hallucination={hall:.2f} (max {self.max_hallucination:.2f})"
            )
        else:
            reason = (
                f"faithfulness gap (coverage={coverage:.2f}, hallucination={hall:.2f}):\n"
                + "\n".join(gap_bullets)
                + "\nRewrite so that every mechanism claim is directly supported by the sentence "
                + "or the provided kg_evidence; remove or hedge any claim not grounded in evidence."
            )

        return CritiqueResult(
            passed=passed,
            score=score,
            reason=reason,
            detail={
                "coverage_ratio": coverage,
                "hallucination_rate": hall,
                "partial_support_rate": float(metric.get("partial_support_rate", 0.0)),
                "consistency_score": float(metric.get("consistency_score", 0.0)),
                "supported_claims_n": len(metric.get("supported_claims", []) or []),
                "partial_supported_claims_n": len(partial),
                "unsupported_claims_n": len(unsupported),
            },
        )


class JudgeCritic(Critic):
    """基于 evaluate.judge.qwen_judge 的 LLM-as-judge 评估。

    通过 `mechanism_overall_score >= min_overall_score` 判断是否达标。
    feedback 是 judge 返回的 short_rationale + mechanism_gaps 拼接。
    """

    name = "judge"

    def __init__(
        self,
        api_key: str,
        model_id: str = "qwen-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        min_overall_score: float = 7.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = False,
    ):
        from evaluate.judge.qwen_judge import QwenMaxJudge  # 延迟 import
        self._judge = QwenMaxJudge(
            api_key=api_key,
            model_id=model_id,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )
        self.min_overall_score = float(min_overall_score)

    def evaluate(self, sample: Dict[str, Any], output: str) -> CritiqueResult:
        judge_input = {
            "sentence": sample.get("sentence", ""),
            "query_group": sample.get("query_group", "") or sample.get("queries", ""),
            "kg_evidence": sample.get("kg_evidence", ""),
            "reasoning": output,
        }
        result = self._judge.judge_sample(judge_input)
        overall = float(result.get("mechanism_overall_score", 0.0))
        passed = overall >= self.min_overall_score

        gaps = result.get("mechanism_gaps", []) or []
        rationale = str(result.get("judge_short_rationale", "")).strip()
        gap_bullets = "\n".join(f"- {g}" for g in gaps if str(g).strip())
        reason_parts: List[str] = []
        reason_parts.append(
            f"judge mechanism_overall_score={overall:.1f} (min {self.min_overall_score:.1f})"
        )
        if rationale:
            reason_parts.append(f"rationale: {rationale}")
        if gap_bullets:
            reason_parts.append(f"gaps:\n{gap_bullets}")
        reason_parts.append(
            "Rewrite the mechanism chain so it is complete (precipitant → biological process → object drug/result), "
            "directionally explicit (who affects whom, increase/decrease), and grounded at a verifiable "
            "granularity (specific enzymes/transporters/receptors when supported)."
        )

        return CritiqueResult(
            passed=passed,
            score=overall / 12.0,
            reason="\n".join(reason_parts),
            detail={
                "mechanism_overall_score": overall,
                "mechanism_overall_decision": result.get("mechanism_overall_decision", ""),
                "mechanism_chain_completeness": result.get("mechanism_chain_completeness", 0),
                "mechanism_direction_correctness": result.get("mechanism_direction_correctness", 0),
                "mechanism_granularity": result.get("mechanism_granularity", 0),
                "mechanism_internal_consistency": result.get("mechanism_internal_consistency", 0),
                "uncertainty_calibration": result.get("uncertainty_calibration", 0),
                "clinical_actionability": result.get("clinical_actionability", 0),
            },
        )


# ---------------------------------------------------------------------------
# Loop 主逻辑
# ---------------------------------------------------------------------------


def _build_feedback_message(critiques: List[CritiqueResult]) -> str:
    """把上一轮所有未通过 critic 的反馈拼成一条 user turn 内容。"""
    parts: List[str] = [
        "Your previous answer did not meet quality thresholds. Fix the issues below and produce a REVISED answer in the same format.",
    ]
    for c in critiques:
        if c.passed:
            continue
        parts.append(f"\n[{c.reason}]")
    parts.append(
        "\nProduce the revised answer only. Do NOT explain what you changed; just output the corrected JSON/text."
    )
    return "\n".join(parts)


class AgentLoopReasoner:
    """反思-重试推理循环。

    用法::

        loop = AgentLoopReasoner(
            reasoner=my_reasoner_fn,
            critics=[FaithfulnessCritic(), JudgeCritic(api_key=...)],
            config=AgentLoopConfig(max_rounds=3),
        )
        trace = loop.run(sample=row, initial_messages=messages)
        final_output = trace.final_output
    """

    def __init__(
        self,
        reasoner: ReasonerFn,
        critics: List[Critic],
        config: Optional[AgentLoopConfig] = None,
    ):
        if not callable(reasoner):
            raise ValueError("reasoner must be a callable (messages, round_idx) -> str")
        if not critics:
            raise ValueError("at least one Critic must be provided")
        self.reasoner = reasoner
        self.critics = list(critics)
        self.config = config or AgentLoopConfig()

    def run(
        self,
        sample: Dict[str, Any],
        initial_messages: List[Dict[str, str]],
    ) -> AgentLoopTrace:
        """对单条样本执行完整循环。"""
        trace = AgentLoopTrace()
        messages: List[Dict[str, str]] = [dict(m) for m in initial_messages]
        last_output = ""
        last_score = 0.0
        last_all_pass = False

        for round_idx in range(1, self.config.max_rounds + 1):
            feedback_injected = round_idx > 1
            try:
                output = self.reasoner(messages, round_idx)
            except Exception as exc:
                if self.config.verbose:
                    print(f"[AgentLoop] round {round_idx} generator error: {exc}")
                trace.stopped_reason = "generator_error"
                trace.rounds.append(
                    RoundRecord(
                        round_idx=round_idx,
                        output="",
                        critiques=[{"name": "generator_error", "reason": str(exc), "passed": False, "score": 0.0, "detail": {}}],
                        feedback_injected=feedback_injected,
                    )
                )
                # 保留上一轮 output（可能非空）作为 final
                trace.final_output = last_output
                trace.final_passed = last_all_pass
                trace.final_score = last_score
                return trace

            # 评估
            round_critiques: List[CritiqueResult] = []
            for critic in self.critics:
                try:
                    res = critic.evaluate(sample, output)
                except Exception as exc:
                    res = CritiqueResult(
                        passed=False,
                        score=0.0,
                        reason=f"{critic.name} error: {exc}",
                        detail={"error": str(exc)},
                    )
                round_critiques.append(res)

            all_pass = all(c.passed for c in round_critiques)
            round_score = sum(c.score for c in round_critiques) / max(1, len(round_critiques))
            last_output = output
            last_all_pass = all_pass
            last_score = round_score

            trace.rounds.append(
                RoundRecord(
                    round_idx=round_idx,
                    output=output,
                    critiques=[
                        {"name": c.__class__.__name__.lower().replace("critic", ""), **asdict(res)}
                        for c, res in zip(self.critics, round_critiques)
                    ],
                    feedback_injected=feedback_injected,
                )
            )

            if self.config.verbose:
                summary = " / ".join(
                    f"{c.name}:{'PASS' if r.passed else 'FAIL'}({r.score:.2f})"
                    for c, r in zip(self.critics, round_critiques)
                )
                print(f"[AgentLoop] round {round_idx}/{self.config.max_rounds} -> {summary}")

            if all_pass and self.config.stop_when_all_pass:
                trace.stopped_reason = "all_pass"
                break

            # 未达标 → 若还有下一轮，把 assistant 的 output 与反馈拼进 messages
            if round_idx < self.config.max_rounds:
                messages = messages + [
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": _build_feedback_message(round_critiques)},
                ]
            else:
                trace.stopped_reason = "max_rounds"

        trace.final_output = last_output
        trace.final_passed = last_all_pass
        trace.final_score = last_score
        return trace


# ---------------------------------------------------------------------------
# Trace 序列化
# ---------------------------------------------------------------------------


def trace_to_dict(trace: AgentLoopTrace, keep_full_messages: bool = False) -> Dict[str, Any]:
    """把 trace 转成 JSON 可序列化的 dict。"""
    payload = {
        "final_output": trace.final_output,
        "final_passed": bool(trace.final_passed),
        "final_score": float(trace.final_score),
        "stopped_reason": trace.stopped_reason,
        "n_rounds": len(trace.rounds),
        "rounds": [
            {
                "round_idx": r.round_idx,
                "output": r.output if keep_full_messages else (r.output[:2000] + "…" if len(r.output) > 2000 else r.output),
                "critiques": r.critiques,
                "feedback_injected": r.feedback_injected,
            }
            for r in trace.rounds
        ],
    }
    return payload


def save_trace_batch(traces_by_id: Dict[str, AgentLoopTrace], path: str, keep_full_messages: bool = False) -> None:
    """把一批 trace 落盘。key 用 sample id / index。"""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dumped = {k: trace_to_dict(v, keep_full_messages=keep_full_messages) for k, v in traces_by_id.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dumped, f, ensure_ascii=False, indent=2)
