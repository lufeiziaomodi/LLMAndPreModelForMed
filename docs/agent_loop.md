# Agent Loop：反思-重试推理机制

> 代码入口：[`pipelines/agent_loop.py`](../pipelines/agent_loop.py)
> 集成点：[`inference_llama3_ddi.py`](../inference_llama3_ddi.py)（`--enable_agent_loop`）与 [`apps/run_external_direct_eval.py`](../apps/run_external_direct_eval.py)（YAML `agent_loop.enabled`）

## 1. 动机

无论是**外部大模型直接推理**（qwen-plus / qwen-max）还是**LoRA 微调后的 Llama3 推理**，第一次生成时都可能出现：

- **句外机制编造**：句子里没提到 CYP3A4，但 output 里写了"通过 CYP3A4 抑制代谢"→ faithfulness `hallucination_rate` 升高。
- **机制链断裂**：只说了"存在相互作用"，没串起"施动药 → 生物过程 → 受体药/结果"→ judge `mechanism_chain_completeness` 得 0。
- **方向反转**：把"A 抑制 B"写成"B 抑制 A" → judge `mechanism_direction_correctness` 得 0。
- **粒度不够**：泛泛说"影响代谢"却不指到具体酶 → judge `mechanism_granularity` 得 0-1。

一次生成难以同时满足"事实约束"与"机制深度"两个方向。Agent Loop 的做法是**把评估反馈回注上下文，让模型自我修正**，代价是每条样本多花 1-2 次生成 + 1-N 次评估。

## 2. 循环结构

```
                 ┌───────────────────────┐
initial_messages │ system + user prompt  │
                 └───────────┬───────────┘
                             ▼
                       ┌──────────┐
                 ┌────▶│ generate │──▶ output
                 │     └─────┬────┘
                 │           ▼
                 │    ┌───────────────┐
                 │    │ FaithCritic   │ 本地 evaluate/explanation/faithfulness.py
                 │    │ + (optional)  │
                 │    │   JudgeCritic │ 调 qwen-max
                 │    └──────┬────────┘
                 │           │
                 │      all pass?
                 │       │      │
                 │      yes    no
                 │       │      │
                 │       ▼      ▼
                 │   final  ┌──────────────────────┐
                 │  output  │ append feedback turn │
                 │          │   assistant: <output>│
                 │          │   user: <gap bullets>│
                 └──────────┴──────────────────────┘
                          round < max?  → next iteration
```

每轮记录一条 `RoundRecord`：
- `round_idx`：轮次（1 起）
- `output`：模型该轮输出
- `critiques`：每个 critic 的 `passed / score / reason / detail`
- `feedback_injected`：本轮之前是否注入过上一轮反馈

## 3. 内置 Critic

### 3.1 `FaithfulnessCritic`（本地，无 API 成本）

调 [`evaluate.explanation.faithfulness.compute_faithfulness_for_sample`](../evaluate/explanation/faithfulness.py)：

- **通过条件**：`coverage_ratio >= min_coverage` **且** `hallucination_rate <= max_hallucination`
- 默认阈值：`min_coverage=0.4`、`max_hallucination=0.5`
- 反馈内容：把 `unsupported_claims` 与 `partial_supported_claims` 拼成 bullet list：
  ```
  faithfulness gap (coverage=0.30, hallucination=0.55):
  - UNSUPPORTED by evidence: <claim>
  - PARTIALLY supported (needs tightening): <claim>
  Rewrite so that every mechanism claim is directly supported by the sentence
  or the provided kg_evidence; remove or hedge any claim not grounded in evidence.
  ```

### 3.2 `JudgeCritic`（调 qwen-max，有 API 成本）

调 [`evaluate.judge.qwen_judge.QwenMaxJudge`](../evaluate/judge/qwen_judge.py)：

- **通过条件**：`mechanism_overall_score >= min_overall_score`（0-12 分制，默认 7.0）
- 反馈内容：`judge_short_rationale` + `mechanism_gaps`，末尾附一段修正指令：
  ```
  Rewrite the mechanism chain so it is complete (precipitant → biological process → object drug/result),
  directionally explicit (who affects whom, increase/decrease), and grounded at a verifiable
  granularity (specific enzymes/transporters/receptors when supported).
  ```

**建议**：开发调试期只开 `FaithfulnessCritic`（本地免费）；正式实验时若目标是机制质量，再加 `JudgeCritic`。

## 4. 配置字段

### 4.1 外部直接评估（YAML）

```yaml
agent_loop:
  enabled: true
  max_rounds: 3                          # 含初始生成，共 N 轮
  faithfulness_min_coverage: 0.4
  faithfulness_max_hallucination: 0.5
  use_judge_critic: false                # 是否加 judge critic
  judge_min_overall_score: 7.0           # 0-12 分制
  keep_full_messages_in_trace: false     # false 时 trace 里 output 超过 2000 字符会截断
  verbose: true
```

### 4.2 LoRA 推理（CLI）

```
--enable_agent_loop
--agent_max_rounds 3
--faith_min_coverage 0.4
--faith_max_hallucination 0.5
--use_judge_critic
--judge_min_overall_score 7.0
--judge_model_id qwen-max
--judge_api_key <or set DASHSCOPE_API_KEY env>
```

## 5. 落盘 trace 结构

Agent Loop 会为每条样本落一份 trace（key 是样本 index）：

```json
{
  "3": {
    "final_output": "<最终 output 文本>",
    "final_passed": true,
    "final_score": 0.72,
    "stopped_reason": "all_pass",
    "n_rounds": 2,
    "rounds": [
      {
        "round_idx": 1,
        "output": "…",
        "critiques": [
          {"name": "faithfulness", "passed": false, "score": 0.55,
           "reason": "faithfulness gap …",
           "detail": {"coverage_ratio": 0.30, "hallucination_rate": 0.55, ...}}
        ],
        "feedback_injected": false
      },
      {
        "round_idx": 2,
        "output": "…修正后…",
        "critiques": [
          {"name": "faithfulness", "passed": true, "score": 0.85, ...}
        ],
        "feedback_injected": true
      }
    ]
  }
}
```

`stopped_reason` 取值：
- `all_pass` — 所有 critic 都通过，提前退出
- `max_rounds` — 走满 `max_rounds` 仍未全通过
- `generator_error` — 生成器抛异常（写在最后一轮的 critique 里）

`out_entry["agent_loop"]`（写在推理结果的每条样本里）汇总一下 trace 关键指标，便于事后统计"平均几轮收敛 / 通过率"。

## 6. 阈值调优建议

| 现象 | 调整方向 |
|---|---|
| 大多数样本第 1 轮就 pass，几乎没触发反思 | `faith_min_coverage` 提到 0.6+，`faith_max_hallucination` 降到 0.3；或者启用 judge critic |
| 大多数样本走满 max_rounds 仍不通过 | 阈值放松（`min_coverage=0.3`, `max_hallucination=0.6`）；或看 trace 里 unsupported claim 是不是明显在 kg_evidence 里但没被算作 supported → 说明 faithfulness 的 tokenizer 或 stopwords 需要迭代 |
| Judge critic 通过率长期偏低 | 说明模型本身机制粒度不够，光靠反思循环解决不了 —— 考虑先做 KG evidence summary 规则化，或换更强的基座 |
| Judge critic 通过率长期偏高 | 说明 judge prompt 太宽松，收紧 rubric 或提高 `judge_min_overall_score` |

## 7. 成本与延迟粗算

以 test 集 1000 条样本为例：

| 配置 | 单条平均生成次数 | 单条平均 API 调用 | 大致费用（qwen-max） |
|---|---|---|---|
| 关闭 agent loop | 1 | 1（外部直评）/ 0（本地推理） | ~$3 |
| 只开 FaithCritic，max_rounds=3，通过率 60% | 1.6 | 1.6 | ~$5 |
| Faith+Judge，max_rounds=3，通过率 40% | 2.0 | 2.0 + 2.0（judge）= 4.0 | ~$15 |

（数量级估计，实际取决于 token 长度）

## 8. 已知局限

1. **模型对反馈的可接受度取决于基座能力**：LoRA 微调后的 Llama3-8B 有时会"照着 gap 抄"而不真的改机制，需要看 trace 判断。
2. **critic 与生成器耦合太紧**：目前 faithfulness critic 用的是同一份 `compute_faithfulness_for_sample`，会带上句证 tokenizer 的偏见；未来可以引入独立的 rule-based 或第三方 LLM critic 做交叉验证。
3. **Trace 会显著增大 report 目录体积**：如果 `keep_full_messages_in_trace=true` 且样本数多，单个 `agent_trace_*.json` 会到几百 MB。建议正式实验保持默认 false。
