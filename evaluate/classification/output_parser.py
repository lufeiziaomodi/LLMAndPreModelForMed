import re
from typing import List, Tuple


DDI_TYPES = ["effect", "mechanism", "advise", "int", "false"]


def extract_prediction(text: str, ddi_types: List[str] = DDI_TYPES) -> Tuple[str, str]:
    relation = "false"
    reasoning = ""

    match = re.search(r"\{(.*)\}", text, re.S)
    content = match.group(1) if match else text
    parts = content.split(",", 1)

    if parts:
        candidate = parts[0].strip().lower()
        if candidate in ddi_types:
            relation = candidate
        else:
            for ddi_type in ddi_types:
                if ddi_type in candidate:
                    relation = ddi_type
                    break

    if len(parts) > 1:
        reasoning = parts[1].strip().strip("}")

    if not reasoning:
        reasoning = text.strip()

    reasoning = reasoning.strip().strip("{}").strip()

    if relation not in ddi_types:
        lower_text = text.lower()
        for ddi_type in ddi_types:
            if ddi_type in lower_text:
                relation = ddi_type
                break

    return relation, reasoning
