import re
from typing import Any, Dict, List


PAIR_CONNECTORS = ("->", "=>", "→")


def get_query_group_text(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    value = data.get("query_group", "")
    if str(value or "").strip():
        return str(value)
    return str(data.get("queries", "") or "")


def _strip_line_prefix(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"^[-*]\s*", "", s)
    return s.strip()


def expand_query_pairs(query_text: Any) -> List[str]:
    pairs: List[str] = []
    seen = set()
    for raw_line in str(query_text or "").replace("\r", "\n").split("\n"):
        line = _strip_line_prefix(raw_line)
        if not line:
            continue

        connector = next((item for item in PAIR_CONNECTORS if item in line), "")
        if not connector:
            continue
        left, right = line.split(connector, 1)
        left = _strip_line_prefix(left)
        right = _strip_line_prefix(right)
        if not left or not right:
            continue

        if right.startswith("{") and right.endswith("}"):
            candidates = [part.strip() for part in right[1:-1].split(",")]
        else:
            candidates = [right]

        for candidate in candidates:
            if not candidate:
                continue
            pair = f"{left} -> {candidate}"
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs
