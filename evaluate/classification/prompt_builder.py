import re
from typing import Tuple


def extract_drugs(text: str) -> Tuple[str, str, str]:
    e1_match = re.search(r"<e1>(.*?)</e1>", text)
    e2_match = re.search(r"<e2>(.*?)</e2>", text)
    drug1 = e1_match.group(1) if e1_match else ""
    drug2 = e2_match.group(1) if e2_match else ""
    clean_text = re.sub(r"</?e[12]>", "", text)
    return drug1, drug2, clean_text


def create_prompt_shot(drug1: str, drug2: str, context: str) -> str:
    return f"""You are performing Drug-Drug Interaction (DDI) relation classification.
Classify ONLY the relation between <e1> and <e2> into one of: effect, mechanism, advise, int, false.

Rules:
1. Only evaluate <e1> and <e2>.
2. If no direct relation is stated for the pair, label is false.
3. Do not hallucinate evidence.

Output format:
{{relation_type, \"reasoning chain\": \"brief justification\"}}

Sentence:
{context}

Drug 1: {drug1}
Drug 2: {drug2}
"""
