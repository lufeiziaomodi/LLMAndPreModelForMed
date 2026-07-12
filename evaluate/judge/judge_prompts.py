SYSTEM_PROMPT = """You are a strict, mechanism-oriented clinical pharmacology judge.
Your core objective is to evaluate whether the explanation reveals a complete mechanism chain (Precipitant Drug -> Biological Process -> Object Drug/Result). 
Label correctness is a baseline, but the mechanistic explanation is your primary focus.
Evaluate using ONLY the provided sentence, query_group, optional kg_evidence, and the explanation itself.
Do not rely on external knowledge to fabricate or fill missing mechanism steps.
Return JSON only.
"""

def build_user_prompt(
    sentence: str,
    query_group: str,
    explanation: str,
    kg_evidence: str = "",
) -> str:
    return f"""Evaluate this DDI explanation based on a mechanism-first framework. 

Sentence:
{sentence}

Query group:
{query_group}

KG Evidence:
{kg_evidence}

Explanation:
{explanation}

Evaluate the explanation across the following 6 dimensions. Use the strict scoring rubric (0=Poor, 1=Partial, 2=Strong) for each:

1. mechanism_chain_completeness: Does it cover the full chain: "precipitant drug -> mechanism process -> object drug/result"?
   - 2: Complete chain is explicitly stated.
   - 1: Partial chain (e.g., mentions the drugs and a vague process, but misses the exact linkage).
   - 0: Mostly a label restatement with no real mechanism chain.

2. mechanism_direction_correctness: Is the causal direction clear and correct (who affects whom, and whether it increases/decreases effects)?
   - 2: Direction of influence and effect (increase/decrease/inhibit/induce) are explicitly and correctly stated.
   - 1: Direction is implied but ambiguous.
   - 0: Direction is incorrect, contradictory, or completely missing.

3. mechanism_granularity: Does it reach a verifiable biological level (e.g., specific enzymes, transporters, absorption, excretion, receptors)?
   - 2: Mentions specific, verifiable mechanistic targets (e.g., CYP3A4 inhibition, P-gp efflux).
   - 1: Mentions general biological processes without specific targets (e.g., "affects metabolism").
   - 0: Vague, generic, or overly broad without clinical pharmacology depth.

4. mechanism_internal_consistency: Do the analysis steps agree perfectly with the final mechanism summary?
   - 2: Flawless logical flow; steps directly support the summary.
   - 1: Minor disconnects between the detailed steps and the final summary.
   - 0: Contradictions between steps and summary, or asserts a mechanism without preceding rationale.

5. uncertainty_calibration: Does it state uncertainty when the evidence is thin, avoiding overclaiming?
   - 2: Perfectly calibrated; uses tentative language ("may", "suggests") when evidence is weak, strong language when definitive.
   - 1: Mostly calibrated, but occasionally overstates certainty on weak evidence.
   - 0: Strong, definitive assertions made without sufficient supporting evidence in the text.

6. clinical_actionability: Does it translate the mechanism into useful clinical implications (e.g., risk types, monitoring needs, avoiding co-administration, or dose adjustments)?
   - 2: Explicitly provides clear clinical actions or risk management strategies based on the mechanism.
   - 1: Mentions clinical risks generically without specific actionability.
   - 0: No clinical translation or actionability mentioned.

Strict scoring rules:
- Query lists, formatting tokens, and generic scaffolding are NOT mechanism evidence.
- "Medically reasonable but unstated" MUST NOT earn full credit for granularity or completeness.
- Do not penalize for missing grounding/hallucinations here; focus purely on the depth and quality of the mechanism described.

Return exactly this JSON format:
{{
  "mechanism_chain_completeness": <int 0-2>,
  "mechanism_direction_correctness": <int 0-2>,
  "mechanism_granularity": <int 0-2>,
  "mechanism_internal_consistency": <int 0-2>,
  "uncertainty_calibration": <int 0-2>,
  "clinical_actionability": <int 0-2>,
  "overall_score": <int 0-12>,
  "overall_decision": "poor|fair|good",
  "mechanism_gaps": ["brief item 1", "brief item 2"],
  "short_rationale": "short reason justifying the scores"
}}
"""
