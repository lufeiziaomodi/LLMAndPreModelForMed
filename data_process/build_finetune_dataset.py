"""
构造微调数据集：从 mechanism.json 和 PrimeKG 构建 MedGemma 训练数据

步骤：
1. 读取 mechanism.json
2. 查询 PrimeKG 获取药物相关的 protein/gene
3. 构造输入格式
4. 生成 prompt 和输出示例
"""

import json
import random
import argparse
import re
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Allow running this file directly via: python data_process/build_finetune_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prime_kg_utils.select import GraphQuery
from pipelines.query_utils import get_query_group_text

# 五种标签对应的源数据文件和标准标签名
LABEL_CONFIG = {
    "Mechanism": "mechanism.json",
    "Effect":    "effect.json",
    "Advice":    "advise.json",
    "Int":       "int.json",
    "False":     "false.json",
}

# 全局停用节点：高连接度且常见的非特异性载体/管家基因
STOP_NODES = {
    "ALB", "ORM1", "ORM2", "TF", "HBB", "HBA1",  # Carrier proteins
    "ACTB", "GAPDH", "UBC",  # Housekeeping / high-degree
}


class DrugKGQuery:
    """查询药物在 PrimeKG 中的相关信息"""

    def __init__(self, graph_path: str = "data/primekg_graph.pkl", mapping_path: str = "data/drug_name_map.json"):
        self.gq = GraphQuery(graph_path)
        self.mapping_path = Path(mapping_path)
        if not self.mapping_path.is_absolute():
            self.mapping_path = (PROJECT_ROOT / self.mapping_path).resolve()
        self._user_name_map: Dict[str, Optional[str]] = {}
        self._user_name_map_raw: Dict[str, Optional[str]] = {}
        self._kg_name_lookup: Dict[str, str] = {}
        self._kg_name_keys: List[str] = []
        self._drug_resolution_cache: Dict[str, Optional[str]] = {}
        self._drug_targets_cache: Dict[str, Dict[str, Any]] = {}
        self._node_degree_cache: Dict[str, int] = {}
        self._load_user_map()
        self._build_kg_name_lookup()

    def _build_kg_name_lookup(self) -> None:
        """Build lowercase lookup tables for PrimeKG node names."""
        lookup: Dict[str, str] = {}
        keys: List[str] = []
        for n in self.gq.g.name_to_indices.keys():
            norm = n.lower().strip()
            if not norm or norm in lookup:
                continue
            lookup[norm] = n
            keys.append(norm)
        self._kg_name_lookup = lookup
        self._kg_name_keys = keys

    def _load_user_map(self) -> None:
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            # 兼容两种格式：
            # 1) {"Felbatol": "felbamate", "X": null}
            # 2) [{"source": "Felbatol", "target": "felbamate"}, ...]
            raw_map: Dict[str, Optional[str]] = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(k, str):
                        continue
                    val = v if (isinstance(v, str) and v.strip()) else None
                    # 与源名称相同（仅大小写差异）的值视为未对齐，统一置为 None。
                    if isinstance(val, str) and val.casefold().strip() == k.casefold().strip():
                        val = None
                    raw_map[k.strip()] = val
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    src = item.get("source")
                    tgt = item.get("target")
                    if not isinstance(src, str):
                        continue
                    val = tgt if (isinstance(tgt, str) and tgt.strip()) else None
                    if isinstance(val, str) and val.casefold().strip() == src.casefold().strip():
                        val = None
                    raw_map[src.strip()] = val

            # 大小写无关去重，避免重复条目；优先保留非空值（人工已对齐）
            dedup_raw: Dict[str, Optional[str]] = {}
            seen_keys: Dict[str, str] = {}
            for k, v in raw_map.items():
                norm = k.casefold().strip()
                if not norm:
                    continue
                if norm not in seen_keys:
                    seen_keys[norm] = k
                    dedup_raw[k] = v
                else:
                    existing_key = seen_keys[norm]
                    if dedup_raw.get(existing_key) is None and v is not None:
                        dedup_raw[existing_key] = v

            self._user_name_map_raw = dedup_raw
            # 存储为小写键，便于大小写不敏感匹配
            self._user_name_map = {
                k.casefold().strip(): v
                for k, v in self._user_name_map_raw.items()
                if isinstance(k, str) and k.strip()
            }
        else:
            self._user_name_map = {}
            self._user_name_map_raw = {}

    def _mapped_drug_name(self, drug_name: str) -> Optional[str]:
        key = drug_name.casefold().strip()
        mapped = self._user_name_map.get(key)
        return mapped if (isinstance(mapped, str) and mapped.strip()) else None

    def _persist_user_map_file(self) -> None:
        try:
            # 持久化前再做一次大小写无关去重
            dedup_raw: Dict[str, Optional[str]] = {}
            seen_keys: Dict[str, str] = {}
            for k, v in self._user_name_map_raw.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                norm = k.casefold().strip()
                if norm not in seen_keys:
                    seen_keys[norm] = k
                    dedup_raw[k] = v if (isinstance(v, str) and v.strip()) else None
                else:
                    existing_key = seen_keys[norm]
                    if dedup_raw.get(existing_key) is None and isinstance(v, str) and v.strip():
                        dedup_raw[existing_key] = v.strip()
            self._user_name_map_raw = dedup_raw

            with open(self.mapping_path, "w", encoding="utf-8") as f:
                # 文件里保留原始药物名 -> 对齐名（或 null）药物对，便于外部人工/程序回填
                json.dump(self._user_name_map_raw, f, ensure_ascii=False, indent=2)
        except Exception:
            # 写入失败不阻塞主流程
            pass

    def _persist_mapping(self, drug_name: str, mapped_to: Optional[str]) -> None:
        """持久化映射，便于后续运行复用人工或自动的别名。"""
        raw_key = (drug_name or "").strip()
        norm_key = raw_key.casefold()
        if not raw_key:
            return

        mapped_value = mapped_to.strip() if isinstance(mapped_to, str) and mapped_to.strip() else None
        has_existing = norm_key in self._user_name_map
        current = self._user_name_map.get(norm_key)
        if isinstance(current, str) and not current.strip():
            current = None

        # 仅当该键已存在且值未变化时跳过；
        # 对于“新键 + mapped_value=None”必须写入（用于未匹配待对齐）
        if has_existing and current == mapped_value:
            return

        # 优先复用已有大小写键，避免同名大小写变体重复
        existing_raw_key = None
        for k in self._user_name_map_raw.keys():
            if k.casefold().strip() == norm_key:
                existing_raw_key = k
                break
        key_to_use = existing_raw_key if existing_raw_key is not None else raw_key

        self._user_name_map[norm_key] = mapped_value
        self._user_name_map_raw[key_to_use] = mapped_value
        self._persist_user_map_file()

    def record_unmatched(self, drug_name: str) -> None:
        key = drug_name.casefold().strip()
        if not key or key in self._user_name_map:
            return
        # 记录未匹配药物以便后续人工维护
        self._persist_mapping(drug_name, None)

    def _normalize_variants(self, drug_name: str) -> List[str]:
        variants = set()
        base = drug_name.lower().strip()
        variants.add(base)
        # 去除常见符号
        stripped = base.replace("-", " ").replace("/", " ")
        variants.add(stripped)
        variants.add(stripped.replace(" ", ""))
        return list(variants)

    def _has_node(self, name: str) -> bool:
        return self._resolve_name_ci(name) is not None

    def _resolve_name_ci(self, name: str) -> Optional[str]:
        """大小写不敏感地解析图谱中的节点名称，返回实际存储的名称。"""
        target = name.lower().strip()
        if not target:
            return None
        return self._kg_name_lookup.get(target)

    def _node_degree(self, name: str) -> int:
        """返回节点度数（出度+入度）用于过滤高噪声节点。"""
        if name in self._node_degree_cache:
            return self._node_degree_cache[name]
        indices = self.gq.g.name_to_indices.get(name, [])
        degree = 0
        for idx in indices:
            degree += len(self.gq.g.out_adj.get(idx, []))
            degree += len(self.gq.g.in_adj.get(idx, []))
        self._node_degree_cache[name] = degree
        return degree

    def _role_from_display(self, relation: str, default: str = "Gene/protein") -> str:
        rel = relation.lower()
        if "enzyme" in rel:
            return "Enzyme"
        if "transporter" in rel:
            return "Transporter"
        if "target" in rel:
            return "Target"
        if "carrier" in rel:
            return "Carrier"
        return default

    def find_drug_in_kg(self, drug_name: str) -> Optional[str]:
        # 显式先转小写再进行图谱检索
        drug_lower = drug_name.lower().strip()
        if not drug_lower:
            return None

        if drug_lower in self._drug_resolution_cache:
            return self._drug_resolution_cache[drug_lower]

        # 1) 显式用户映射（人工或自动维护的别名）
        mapped = self._mapped_drug_name(drug_name)
        if mapped:
            resolved = self._resolve_name_ci(mapped)
            if resolved:
                self._drug_resolution_cache[drug_lower] = resolved
                return resolved

        # 2) 精确匹配原始名称
        resolved = self._resolve_name_ci(drug_name)
        if resolved:
            self._drug_resolution_cache[drug_lower] = resolved
            return resolved

        # 3) 在所有节点中搜索（大小写不敏感）
        resolved = self._kg_name_lookup.get(drug_lower)
        if resolved:
            self._drug_resolution_cache[drug_lower] = resolved
            return resolved

        # 4) 部分匹配（包含或被包含）
        for name_lower in self._kg_name_keys:
            if len(drug_lower) >= 3 and len(name_lower) >= 3:
                if drug_lower in name_lower or name_lower in drug_lower:
                    resolved = self._kg_name_lookup.get(name_lower)
                    if resolved:
                        self._drug_resolution_cache[drug_lower] = resolved
                        return resolved

        # 5) 名称变体/映射兜底
        variants = self._normalize_variants(drug_name)
        for variant in variants:
            # 优先使用映射值
            mapped_name = self._user_name_map.get(variant)
            candidate = mapped_name if mapped_name else variant
            resolved = self._resolve_name_ci(candidate)
            if resolved:
                self._drug_resolution_cache[drug_lower] = resolved
                return resolved

        self._drug_resolution_cache[drug_lower] = None
        return None

    def get_drug_targets(self, drug_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """获取药物相关的 protein/gene 信息"""
        matched_name = self.find_drug_in_kg(drug_name)
        if not matched_name:
            self.record_unmatched(drug_name)
            return {
                "drug_name": drug_name,
                "targets": [],
                "matched": False
            }

        targets = []
        seen = set()

        def _collect_neighbors(name: str):
            nb_out = self.gq.neighbors(name, direction="out", relation=None)
            nb_in = self.gq.neighbors(name, direction="in", relation=None)
            return nb_out + nb_in

        try:
            neighbors = _collect_neighbors(matched_name)
        except Exception as e:
            return {
                "drug_name": drug_name,
                "targets": [],
                "matched": False,
                "error": str(e)
            }

        if matched_name in self._drug_targets_cache:
            return json.loads(json.dumps(self._drug_targets_cache[matched_name], ensure_ascii=False))

        for nbr in neighbors:
            if "to_type" in nbr:
                nbr_type = nbr.get("to_type", "").lower()
                nbr_name = nbr.get("to", "")
                relation = nbr.get("relation", "")
            elif "from_type" in nbr:
                nbr_type = nbr.get("from_type", "").lower()
                nbr_name = nbr.get("from", "")
                relation = nbr.get("relation", "")
            else:
                continue

            if nbr_type in ("protein", "gene", "gene/protein"):
                key = (nbr_name.lower(), nbr_type, relation)
                if key not in seen:
                    seen.add(key)
                    targets.append({
                        "name": nbr_name,
                        "type": nbr_type.capitalize(),
                        "relation": relation,
                        "display_relation": relation,
                        "role": self._role_from_display(relation)
                    })

        result = {
            "drug_name": drug_name,
            "matched_name": matched_name if matched_name != drug_name else None,
            "targets": targets,
            "matched": True
        }
        self._drug_targets_cache[matched_name] = result
        return json.loads(json.dumps(result, ensure_ascii=False))

    def format_drug_profile(self, drug_name: str) -> str:
        """格式化药物信息为字符串，用于输入"""
        profile = self.get_drug_targets(drug_name)
        targets = profile["targets"]
        if not targets:
            return f"{drug_name} (Targets: [])"

        target_strs = []
        for t in targets:
            target_name = t["name"]
            role = t.get("role") or self._role_from_display(t.get("relation", ""), "Gene/protein")
            target_strs.append(f"{target_name}({role})")

        return f"{drug_name} (Targets: [{', '.join(target_strs)}])"


def load_mechanism_data(json_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载 mechanism.json 数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if limit:
        return data[:limit]
    return data


def _make_pending_record(text: str, e1: str, e2: str, gold_label: str, source_file: str) -> Dict[str, Any]:
    return {
        "text": text,
        "e1": e1,
        "e2": e2,
        "gold_label": gold_label,
        "source_file": source_file,
    }


def _resolve_ablation_mode(mode: str) -> Dict[str, Any]:
    mode = (mode or "full").strip().lower()
    alias = {
        "classification_with_kg": "label_only_with_kg",
        "classification_without_kg": "label_only_without_kg",
        "explanation_with_kg": "reasoning_without_label_with_kg",
        "explanation_without_kg": "reasoning_without_kg",
    }
    mode = alias.get(mode, mode)
    mapping = {
        "full": {
            "include_kg_evidence": True,
            "include_gold_label_in_input": True,
            "output_mode": "reasoning",
        },
        "label_only_with_kg": {
            "include_kg_evidence": True,
            "include_gold_label_in_input": False,
            "output_mode": "label",
        },
        "reasoning_without_kg": {
            "include_kg_evidence": False,
            "include_gold_label_in_input": False,
            "output_mode": "reasoning",
        },
        "label_only_without_kg": {
            "include_kg_evidence": False,
            "include_gold_label_in_input": False,
            "output_mode": "label",
        },
        "reasoning_without_label_with_kg": {
            "include_kg_evidence": True,
            "include_gold_label_in_input": False,
            "output_mode": "reasoning",
        },
    }
    if mode not in mapping:
        raise ValueError(
            "--ablation_mode 仅支持 full, label_only_with_kg, reasoning_without_kg, label_only_without_kg, reasoning_without_label_with_kg, classification_with_kg, classification_without_kg, explanation_with_kg, explanation_without_kg"
        )
    return mapping[mode]


def generate_output_template(
    text: str,
    e1: str,
    e2: str,
    drug_a_targets: List[Dict],
    drug_b_targets: List[Dict]
) -> str:
    """
    生成输出模板（示例格式）
    
    实际使用时，这个应该由 MedGemma 模型生成，这里只是提供一个模板示例
    """
    # 找出共同的目标（例如共同的酶）
    drug_a_target_names = {t["name"].lower() for t in drug_a_targets}
    drug_b_target_names = {t["name"].lower() for t in drug_b_targets}
    common_targets = drug_a_target_names & drug_b_target_names
    
    # 构建模板
    template = "Mechanism Analysis: "
    
    if common_targets:
        common_list = list(common_targets)
        template += f"The sentence indicates a drug interaction. From the knowledge graph, both {e1} and {e2} interact with "
        if len(common_list) == 1:
            template += f"the enzyme/protein **{common_list[0]}**. "
        else:
            template += f"the enzymes/proteins **{', '.join(common_list)}**. "
        template += f"Therefore, the interaction mechanism involves these shared targets."
    else:
        template += f"The sentence describes an interaction between {e1} and {e2}. "
        template += f"From the knowledge graph, {e1} targets {', '.join([t['name'] for t in drug_a_targets[:3]])} "
        template += f"while {e2} targets {', '.join([t['name'] for t in drug_b_targets[:3]])}. "
        template += "The mechanism may involve indirect pathways or other biological processes."
    
    return template


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_mechanism_focus(text: str) -> Dict[str, Any]:
    lower = _normalize_text(text)
    focus_items: List[str] = []
    preferred_slots = set()

    if any(k in lower for k in ("renal clearance", "tubular secretion", "base-secreting", "secretion", "excretion", "clearance")):
        focus_items.append("renal tubular secretion / clearance")
        preferred_slots.add("transporter")
    if any(k in lower for k in ("metabolism", "metabol", "cyp", "inhibition", "induction", "substrate")):
        focus_items.append("metabolism / enzyme-mediated interaction")
        preferred_slots.add("enzyme")
    if any(k in lower for k in ("absorption", "bioavailability", "chelation", "gut")):
        focus_items.append("absorption / disposition")
        preferred_slots.add("transporter")
    if any(k in lower for k in ("qt", "arrhythm", "torsades", "brady", "synerg", "depression", "response", "toxicity")):
        focus_items.append("pharmacodynamic effect / target convergence")
        preferred_slots.add("target")
    if any(k in lower for k in ("monitor", "caution", "contraindicated", "avoid")):
        focus_items.append("clinical management / monitoring")

    if not focus_items:
        focus_items.append("general DDI mechanism exploration")
    if not preferred_slots:
        preferred_slots = {"transporter", "enzyme", "target"}

    return {
        "items": focus_items,
        "preferred_slots": preferred_slots,
    }


def _target_slot(target: Dict[str, Any]) -> str:
    role = str(target.get("role", "") or "").lower()
    relation = str(target.get("relation", "") or "").lower()
    name = str(target.get("name", "") or "").upper()

    if "transporter" in role or name.startswith(("SLC", "ABC", "ABCC", "ABCG", "SLCO")):
        return "transporter"
    if "enzyme" in role or name.startswith(("CYP", "UGT", "FMO", "AOX", "MAO", "CES")):
        return "enzyme"
    if "target" in role:
        return "target"
    if "carrier" in role:
        return "carrier"
    if "target" in relation:
        return "target"
    if "enzyme" in relation:
        return "enzyme"
    if "transporter" in relation:
        return "transporter"
    return "gene_protein"


def _is_reliable_mechanism_target(target: Dict[str, Any]) -> bool:
    slot = _target_slot(target)
    relation = str(target.get("relation", "") or "").lower()
    if slot in {"transporter", "enzyme", "target"}:
        return True
    if slot == "carrier":
        return False
    if "ppi" in relation:
        return False
    return False


def _score_target_for_sentence(
    target: Dict[str, Any],
    focus: Dict[str, Any],
    kg_query: DrugKGQuery,
) -> float:
    slot = _target_slot(target)
    relation = str(target.get("relation", "") or "").lower()
    display = str(target.get("display_relation", "") or "").lower()
    name = str(target.get("name", "") or "")
    name_upper = name.upper()

    score = {
        "transporter": 7.0,
        "enzyme": 6.0,
        "target": 5.0,
        "carrier": 1.0,
        "gene_protein": -2.0,
    }.get(slot, 0.0)

    if not _is_reliable_mechanism_target(target):
        score -= 4.0
    if "ppi" in relation:
        score -= 3.0
    if "associated" in relation:
        score -= 1.0

    if slot in focus["preferred_slots"]:
        score += 3.0

    if slot == "transporter" and name_upper.startswith(("SLC22", "SLC47", "ABCB", "ABCC", "ABCG", "SLCO")):
        score += 2.5
    if slot == "enzyme" and name_upper.startswith(("CYP", "UGT", "FMO")):
        score += 1.8
    if slot == "target" and any(k in name_upper for k in ("ADRA", "SCN", "KCN", "HRH", "ACHE", "OPR")):
        score += 1.2

    if "transporter" in display:
        score += 1.0
    if "enzyme" in display:
        score += 0.8
    if "target" in display:
        score += 0.6

    degree = kg_query._node_degree(name) if name else 0
    if degree > 200:
        score -= min(3.0, degree / 300.0)

    return score


def _rank_targets_for_sentence(
    drug_name: str,
    targets: List[Dict[str, Any]],
    focus: Dict[str, Any],
    kg_query: DrugKGQuery,
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    seen = set()
    for t in targets:
        name = str(t.get("name", "") or "")
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        slot = _target_slot(t)
        scored = dict(t)
        scored["slot"] = slot
        scored["score"] = _score_target_for_sentence(t, focus=focus, kg_query=kg_query)
        if _is_reliable_mechanism_target(t):
            ranked.append(scored)

    ranked.sort(
        key=lambda x: (
            -float(x.get("score", 0.0)),
            x.get("slot", ""),
            str(x.get("name", "")),
        )
    )
    return ranked[:max_items]


def _group_queries(relations: List[Dict[str, str]]) -> List[str]:
    grouped: Dict[str, List[str]] = {}
    for rel in relations:
        e1 = rel.get("e1", "").strip()
        e2 = rel.get("e2", "").strip()
        if not (e1 and e2):
            continue
        grouped.setdefault(e1, [])
        if e2 not in grouped[e1]:
            grouped[e1].append(e2)

    lines: List[str] = []
    for src, dsts in grouped.items():
        if len(dsts) == 1:
            lines.append(f"- {src} -> {dsts[0]}")
        else:
            lines.append(f"- {src} -> {{{', '.join(dsts)}}}")
    return lines


def _format_anchor_names(targets: List[Dict[str, Any]], max_items: int = 6) -> str:
    names = [str(t.get("name", "")).strip() for t in targets if str(t.get("name", "")).strip()]
    names = names[:max_items]
    if not names:
        return "no reliable mechanism-level KG anchor found"
    return ", ".join(names)


def _pair_bridge_summary(
    e1: str,
    e2: str,
    ranked_targets: Dict[str, List[Dict[str, Any]]],
) -> str:
    left = ranked_targets.get(e1, [])
    right = ranked_targets.get(e2, [])
    left_names = {str(t.get("name", "")).lower(): t for t in left}
    right_names = {str(t.get("name", "")).lower(): t for t in right}
    shared = [left_names[k]["name"] for k in left_names.keys() & right_names.keys()]
    if shared:
        return f"- {e1} -> {e2}: shared anchors [{', '.join(shared[:4])}]"

    left_slots = {str(t.get("slot", "")) for t in left if str(t.get("slot", ""))}
    right_slots = {str(t.get("slot", "")) for t in right if str(t.get("slot", ""))}
    common_slots = [slot for slot in ("transporter", "enzyme", "target") if slot in left_slots & right_slots]
    if common_slots:
        slot = common_slots[0]
        return f"- {e1} -> {e2}: no direct shared anchor; convergent {slot} context"

    if not left and not right:
        return f"- {e1} -> {e2}: no reliable mechanism-level KG anchors on either side"
    if not left:
        return f"- {e1} -> {e2}: no reliable mechanism-level KG anchor found for {e1}"
    if not right:
        return f"- {e1} -> {e2}: no reliable mechanism-level KG anchor found for {e2}"
    return f"- {e1} -> {e2}: no direct shared anchor; weak KG bridge"


def _build_compact_kg_evidence(
    text: str,
    relations: List[Dict[str, str]],
    ranked_targets: Dict[str, List[Dict[str, Any]]],
) -> str:
    focus = _extract_mechanism_focus(text)
    lines: List[str] = []
    lines.append(f"Mechanism focus: {'; '.join(focus['items'])}.")
    lines.append("")
    lines.append("Reliable KG anchors:")

    ordered_drugs: List[str] = []
    for rel in relations:
        for key in ("e1", "e2"):
            drug = rel.get(key, "").strip()
            if drug and drug not in ordered_drugs:
                ordered_drugs.append(drug)

    for drug in ordered_drugs:
        lines.append(f"- {drug}: {_format_anchor_names(ranked_targets.get(drug, []))}")

    lines.append("")
    lines.append("Pair-level bridge evidence:")
    seen_pairs = set()
    grouped_sources: Dict[str, List[str]] = {}
    for rel in relations:
        e1 = rel.get("e1", "").strip()
        e2 = rel.get("e2", "").strip()
        if e1 and e2:
            grouped_sources.setdefault(e1, [])
            if e2 not in grouped_sources[e1]:
                grouped_sources[e1].append(e2)

    emitted_group_hint = False
    if len(grouped_sources) == 1:
        src = next(iter(grouped_sources.keys()))
        dsts = grouped_sources[src]
        src_targets = ranked_targets.get(src, [])
        if not src_targets and len(dsts) >= 2:
            dst_slots: Dict[str, int] = {}
            for dst in dsts:
                for t in ranked_targets.get(dst, []):
                    slot = str(t.get("slot", ""))
                    if slot in {"transporter", "enzyme", "target"}:
                        dst_slots[slot] = dst_slots.get(slot, 0) + 1
            dominant_slot = ""
            if dst_slots:
                dominant_slot = sorted(dst_slots.items(), key=lambda x: (-x[1], x[0]))[0][0]
            if dominant_slot:
                lines.append(
                    f"- {src}: source drug lacks reliable KG anchors; co-mentioned drugs repeatedly expose {dominant_slot} context consistent with the sentence-level mechanism cue."
                )
            else:
                lines.append(
                    f"- {src}: source drug lacks reliable KG anchors; rely on sentence-level mechanism cue plus indirect anchors from co-mentioned drugs."
                )
            emitted_group_hint = True

    for rel in relations:
        e1 = rel.get("e1", "").strip()
        e2 = rel.get("e2", "").strip()
        if not (e1 and e2):
            continue
        key = (e1, e2)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if emitted_group_hint and len(grouped_sources) == 1 and not ranked_targets.get(e1, []):
            continue
        lines.append(_pair_bridge_summary(e1, e2, ranked_targets))

    focus_text = "; ".join(focus["items"])
    if "transporter" in focus["preferred_slots"]:
        hint = "Interpretation hint: prioritize transporter-mediated competition / renal secretion hypotheses and ignore low-signal generic PPI neighbors."
    elif "enzyme" in focus["preferred_slots"]:
        hint = "Interpretation hint: prioritize enzyme-mediated metabolism hypotheses and ignore low-signal generic PPI neighbors."
    elif "target" in focus["preferred_slots"]:
        hint = "Interpretation hint: prioritize target-level / pharmacodynamic convergence and ignore low-signal generic PPI neighbors."
    else:
        hint = f"Interpretation hint: use sentence-first reasoning, and treat KG only as sparse anchors for {focus_text}."
    lines.append("")
    lines.append(hint)
    return "\n".join(lines).strip()


def create_finetune_entry(
    text: str,
    drug_names: List[str],
    relations: List[Dict[str, str]],
    kg_query: DrugKGQuery,
    include_kg_evidence: bool = True,
    include_gold_label_in_input: bool = True,
    output_mode: str = "reasoning",
    gold_label: str = "Mechanism",
) -> Dict[str, Any]:
    """
    创建单条微调数据（支持多个药物），返回结构与金标一致：
    {
        "instruction": ...,
        "input": {
            "sentence": "...",
            "query_group": "- A -> {B, C}"
        },
        "output": "..."  # 默认空，或按需生成模板
    }
    """
    # 1) 预取药物靶点并构建关系级证据
    all_drug_targets: Dict[str, List[Dict[str, Any]]] = {}
    focus = _extract_mechanism_focus(text)

    # 预取所有药物的 targets
    for drug_name in drug_names:
        profile = kg_query.get_drug_targets(drug_name)
        all_drug_targets[drug_name] = profile.get("targets", [])

    ranked_targets: Dict[str, List[Dict[str, Any]]] = {
        drug_name: _rank_targets_for_sentence(
            drug_name,
            all_drug_targets.get(drug_name, []),
            focus=focus,
            kg_query=kg_query,
        )
        for drug_name in drug_names
    }

    # 计算每对药物的共享节点并应用 STOP_NODES 过滤
    pair_shared: Dict[tuple, List[str]] = {}
    for rel in relations:
        e1 = rel.get("e1", "").strip()
        e2 = rel.get("e2", "").strip()
        if not (e1 and e2):
            continue
        t1 = ranked_targets.get(e1, [])
        t2 = ranked_targets.get(e2, [])
        names1 = {t.get("name", "").lower() for t in t1 if t.get("name")}
        names2 = {t.get("name", "").lower() for t in t2 if t.get("name")}
        shared = names1 & names2
        filtered_shared = []
        for n in shared:
            n_upper = n.upper()
            if n_upper in STOP_NODES and len(shared) > 1:
                # 噪声节点且不是唯一路径 -> 丢弃
                continue
            filtered_shared.append(n)
        pair_shared[(e1, e2)] = filtered_shared

    # 计算是否存在任何共享，用于后续 ADME 策略
    shared_union = set()
    for names in pair_shared.values():
        shared_union.update(names)

    def _adme_priority_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tier1 = []
        tier2 = []
        tier3 = []
        for t in targets:
            name = t.get("name", "")
            if not name:
                continue
            upper = name.upper()
            if upper in STOP_NODES:
                continue
            role = (t.get("role") or "").lower()
            if upper.startswith("CYP") or upper.startswith("ABC") or upper.startswith("SLC"):
                tier1.append(t)
            elif role == "target" or ("target" in (t.get("relation") or "").lower()):
                tier2.append(t)
            else:
                tier3.append(t)

        prioritized = tier1 + tier2 + tier3
        if not prioritized:
            return targets[:3]  # 兜底，避免空

        if len(prioritized) > 20:
            prioritized.sort(key=lambda t: kg_query._node_degree(t.get("name", "")))
            prioritized = prioritized[:20]
        return prioritized

    def _targets_for_drug(drug_name: str) -> List[Dict[str, Any]]:
        targets = ranked_targets.get(drug_name, [])
        if not targets:
            return []

        # 参与的共享集合
        shared_for_drug = set()
        for (a, b), names in pair_shared.items():
            if drug_name == a or drug_name == b:
                shared_for_drug.update(names)

        if shared_union:
            filtered = [t for t in targets if t.get("name", "").lower() in shared_for_drug]
            if filtered:
                return filtered
            # 若该药物未落在共享节点上，退回 ADME 优先策略
            return _adme_priority_targets(targets)

        # 无共享：采用 ADME 优先策略
        return _adme_priority_targets(targets)

    # 2) 构建 queries 与 kg_evidence
    query_group_str = "\n".join(_group_queries(relations))
    kg_evidence_str = _build_compact_kg_evidence(
        text=text,
        relations=relations,
        ranked_targets=ranked_targets,
    )

    if include_kg_evidence:
        instruction = (
            "Analyze the biological mechanisms for the specific drug pairs listed in the 'Query group' field, "
            "based on the provided sentence and Knowledge Graph evidence."
        )
    else:
        instruction = (
            "Analyze the biological mechanisms for the specific drug pairs listed in the 'Query group' field, "
            "based on the provided sentence only. Do not use Knowledge Graph evidence."
        )

    output = ""
    if output_mode == "reasoning":
        if include_kg_evidence:
            parts = ["Mechanism Analysis:\n"]
            for idx, rel in enumerate(relations, 1):
                e1 = rel.get("e1", "").strip()
                e2 = rel.get("e2", "").strip()
                if not (e1 and e2):
                    continue
                part = f"{idx}. **{e1} -> {e2}**: "
                shared = pair_shared.get((e1, e2), [])
                if shared:
                    names = [t.get("name") for t in ranked_targets.get(e1, []) if t.get("name", "").lower() in shared]
                    part += f"Shared anchors {', '.join(names)} suggest overlapping mechanism context."
                else:
                    part += "No direct shared anchor; use compact KG evidence to reason over convergent transporter/enzyme/target context."
                parts.append(part)
            if len(parts) > 1:
                output = "\n\n".join(parts)
        else:
            parts = ["Mechanism Analysis:\n"]
            for idx, rel in enumerate(relations, 1):
                e1 = rel.get("e1", "").strip()
                e2 = rel.get("e2", "").strip()
                if not (e1 and e2):
                    continue
                parts.append(
                    f"{idx}. **{e1} -> {e2}**: Use the sentence cue to explain the interaction type directly, then give a short mechanism summary grounded only in the text."
                )
            if len(parts) > 1:
                output = "\n\n".join(parts)
    elif output_mode == "label":
        output = gold_label

    input_payload: Dict[str, Any] = {
        "sentence": text,
        "query_group": query_group_str,
    }
    if include_kg_evidence:
        input_payload["kg_evidence"] = kg_evidence_str
    if include_gold_label_in_input:
        input_payload["gold_label"] = gold_label

    return {
        "instruction": instruction,
        "input": input_payload,
        "output": output
    }


def process_label_data(
    input_json: str,
    output_json: Optional[str],
    kg_query: DrugKGQuery,
    gold_label: str = "Mechanism",
    limit: Optional[int] = None,
    verbose: bool = True,
    include_kg_evidence: bool = True,
    include_gold_label_in_input: bool = True,
    output_mode: str = "reasoning",
    skip_unmatched: bool = True,
    pending_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    处理单个标签的数据文件，生成微调输入条目。
    与原 process_mechanism_data 一致，新增 gold_label 参数。

    - input_json: 输入 JSON 路径（mechanism.json / effect.json 等）
    - output_json: 输出路径，为 None 时不单独写入文件
    - gold_label: 此文件对应的 DDI 标签名称
    - kg_query: DrugKGQuery 实例
    - limit: 限制处理数（测试用）
    - verbose: 是否打印进度
    """
    # 加载数据
    if verbose:
        print(f"加载数据: {input_json}")
    mechanism_data = load_mechanism_data(input_json, limit=limit)  # type: ignore[arg-type]
    
    if verbose:
        print(f"共 {len(mechanism_data)} 条数据")
    
    # 处理每条数据：一个句子对应一个样例，句内可包含多个 query
    finetune_data = []
    processed_count = 0
    match_stats = {"matched": 0, "not_matched": 0, "total_drugs": 0}
    
    for idx, item in enumerate(mechanism_data, 1):
        text = item.get("text", "").strip()
        relations = item.get("relations", [])
        
        if not text or not relations:
            continue
        
        # 去重后的关系对
        unique_rels = []
        seen_pairs = set()
        for rel in relations:
            e1 = rel.get("e1", "").strip()
            e2 = rel.get("e2", "").strip()
            if not (e1 and e2):
                continue
            pair = (e1, e2)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            unique_rels.append(rel)

        if not unique_rels:
            continue

        valid_rels = []
        for rel in unique_rels:
            e1 = rel.get("e1", "").strip()
            e2 = rel.get("e2", "").strip()
            if not (e1 and e2):
                continue

            match_stats["total_drugs"] += 2

            # 检查匹配情况
            for drug in (e1, e2):
                matched_name = kg_query.find_drug_in_kg(drug)
                if matched_name:
                    match_stats["matched"] += 1
                else:
                    match_stats["not_matched"] += 1
                    if verbose:
                        print(f"警告: 未找到药物 '{drug}' 在 PrimeKG 中")
                    kg_query.record_unmatched(drug)

            # 未匹配药物对不写入结果，进入待对齐清单
            if skip_unmatched:
                m1 = kg_query.find_drug_in_kg(e1)
                m2 = kg_query.find_drug_in_kg(e2)
                if not (m1 and m2):
                    if pending_records is not None:
                        pending_records.append(
                            _make_pending_record(
                                text=text,
                                e1=e1,
                                e2=e2,
                                gold_label=gold_label,
                                source_file=Path(input_json).name,
                            )
                        )
                    continue

            valid_rels.append({"e1": e1, "e2": e2})

        if not valid_rels:
            continue

        drug_list = []
        seen_drugs = set()
        for rel in valid_rels:
            for key in ("e1", "e2"):
                d = rel.get(key, "").strip()
                if d and d not in seen_drugs:
                    seen_drugs.add(d)
                    drug_list.append(d)

        try:
            # 一个句子聚合多个关系对
            entry = create_finetune_entry(
                text,
                drug_list,
                valid_rels,
                kg_query,
                include_kg_evidence=include_kg_evidence,
                include_gold_label_in_input=include_gold_label_in_input,
                output_mode=output_mode,
                gold_label=gold_label,
            )
            finetune_data.append(entry)
            processed_count += 1

            if verbose and processed_count % 10 == 0:
                print(f"已处理原始样例 {processed_count} 条...")

        except Exception as e:
            if verbose:
                print(f"处理第 {idx} 条数据时出错: {e}")
            continue
    
    # 打印统计信息
    if verbose:
        print(f"\n匹配统计:")
        print(f"  总药物数: {match_stats['total_drugs']}")
        print(f"  匹配成功: {match_stats['matched']} ({match_stats['matched']/max(match_stats['total_drugs'],1)*100:.1f}%)")
        print(f"  未匹配: {match_stats['not_matched']} ({match_stats['not_matched']/max(match_stats['total_drugs'],1)*100:.1f}%)")
    
    # 保存结果
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(finetune_data, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\n已保存 {len(finetune_data)} 条数据到: {output_json}")
    
    return finetune_data


def main():
    parser = argparse.ArgumentParser(description="Build multi-label DDI finetune input dataset")
    parser.add_argument("--total_samples", type=int, default=1000,
                        help="各标签采样总数")
    parser.add_argument("--ratios", type=str, default="0.3,0.25,0.1,0.05,0.3",
                        help="Mechanism,Effect,Advice,Int,False 的比例，逗号分隔")
    parser.add_argument("--input_suffix", type=str, default="",
                        help="输入文件名后缀，例如 _test 会读取 mechanism_test.json 等")
    parser.add_argument("--use_all_data", action="store_true",
                        help="使用每个标签文件的全部数据，不按比例采样")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认: data/finetune_dataset_input.json）")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "label_only_with_kg",
            "reasoning_without_kg",
            "label_only_without_kg",
            "reasoning_without_label_with_kg",
            "classification_with_kg",
            "classification_without_kg",
            "explanation_with_kg",
            "explanation_without_kg",
        ],
        help="数据集模式：full=KG+标签约束推理；label_only_with_kg=KG+标签；reasoning_without_kg=无KG解释；label_only_without_kg=无KG标签；reasoning_without_label_with_kg=有KG解释(无标签约束)",
    )
    parser.add_argument("--skip_unmatched", action="store_true", default=True,
                        help="未匹配 PrimeKG 的药物对暂不写入输出")
    parser.add_argument("--pending_output", type=str, default=None,
                        help="保存未匹配药物对清单（默认: data/pending_unmatched_pairs.json）")
    parser.add_argument("--retry_pending_json", type=str, default=None,
                        help="仅处理待对齐清单（如 data/pending_unmatched_pairs.json）")
    parser.add_argument("--retry_log_every", type=int, default=100,
                        help="回填模式日志频率（每处理 N 条打印一次，<=0 表示不打印中间进度）")
    parser.add_argument("--append_output", action="store_true",
                        help="将本次新增样例追加到已有输出文件（用于 retry pending）")
    parser.add_argument("--drop_gold_label", action="store_true",
                        help="导出时移除 input.gold_label（用于直接推理测试输入）")
    parser.add_argument("--gold_label_sidecar", type=str, default=None,
                        help="当 --drop_gold_label 启用时，可选保存标签对照文件路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true",
                        help="减少日志输出（等价于 verbose=False）")
    args = parser.parse_args()
    if args.quiet:
        args.verbose = False

    # For test-set construction, hide gold labels by default to avoid leakage in inference input.
    if args.input_suffix == "_test" and not args.drop_gold_label:
        args.drop_gold_label = True
        print("检测到 input_suffix=_test，已自动启用 --drop_gold_label（测试输入不包含 gold_label）")

    random.seed(args.seed)
    ablation = _resolve_ablation_mode(args.ablation_mode)
    project_root = Path(__file__).parent.parent
    graph_path = project_root / "data" / "primekg_graph.pkl"
    mapping_path = project_root / "data" / "drug_name_map.json"
    kg_query = DrugKGQuery(graph_path=str(graph_path), mapping_path=str(mapping_path))
    before_unmatched = sum(1 for v in kg_query._user_name_map_raw.values() if v is None)

    labels = list(LABEL_CONFIG.keys())
    counts: Dict[str, int] = {}
    if not args.use_all_data:
        raw_ratios = [float(r) for r in args.ratios.split(",")]
        assert len(raw_ratios) == len(labels), \
            f"--ratios 需要 {len(labels)} 个比例值，当前只有 {len(raw_ratios)} 个"
        total_ratio = sum(raw_ratios)
        counts = {
            label: round(args.total_samples * (r / total_ratio))
            for label, r in zip(labels, raw_ratios)
        }
        # 修正舍入误差
        diff = args.total_samples - sum(counts.values())
        counts[labels[0]] += diff

    print("=" * 60)
    print("构造多标签微调输入数据集")
    print(f"输入后缀: '{args.input_suffix}'")
    print(f"全量模式: {args.use_all_data}")
    print(f"消融模式: {args.ablation_mode}")
    print(
        f"输入配置: include_kg_evidence={ablation['include_kg_evidence']}, "
        f"include_gold_label_in_input={ablation['include_gold_label_in_input']}"
    )
    print(f"输出配置: {ablation['output_mode']}")
    if args.use_all_data:
        print("采样策略: 使用各标签输入文件全部样本")
    else:
        print(f"目标总样本: {args.total_samples}")
        for label, cnt in counts.items():
            print(f"  {label}: {cnt} 条")
    print("=" * 60)

    all_entries: List[Dict[str, Any]] = []
    pending_records: List[Dict[str, Any]] = []

    if args.retry_pending_json:
        retry_path = Path(args.retry_pending_json)
        if not retry_path.exists():
            raise FileNotFoundError(f"待对齐清单不存在: {retry_path}")
        with open(retry_path, "r", encoding="utf-8") as f:
            retry_items = json.load(f)

        map_total = len(kg_query._user_name_map_raw)
        map_non_null = sum(
            1 for v in kg_query._user_name_map_raw.values()
            if isinstance(v, str) and v.strip()
        )
        print(
            f"[Retry] 映射文件统计: total={map_total}, non_null={map_non_null}, "
            f"path={kg_query.mapping_path}"
        )
        if map_non_null == 0:
            print("[Retry] 警告: 映射文件没有可用对齐值(non_null=0)，回填大概率不会新增成功样例。")

        print(f"\n[Retry] 读取待对齐清单: {retry_path}，共 {len(retry_items)} 条")

        grouped_retry: Dict[tuple, Dict[str, Any]] = {}
        for item in retry_items:
            text = str(item.get("text", "")).strip()
            e1 = str(item.get("e1", "")).strip()
            e2 = str(item.get("e2", "")).strip()
            label = str(item.get("gold_label", "Mechanism")).strip() or "Mechanism"
            src = str(item.get("source_file", "pending.json"))
            if not (text and e1 and e2):
                continue
            key = (text, label, src)
            bucket = grouped_retry.setdefault(
                key,
                {
                    "text": text,
                    "gold_label": label,
                    "source_file": src,
                    "relations": [],
                },
            )
            bucket["relations"].append({"e1": e1, "e2": e2})

        still_pending: List[Dict[str, Any]] = []
        retry_matched = 0
        retry_unmatched = 0
        grouped_values = list(grouped_retry.values())
        for i, group in enumerate(grouped_values, 1):
            text = group["text"]
            label = group["gold_label"]
            src = group["source_file"]
            relations = group["relations"]

            valid_rels = []
            for rel in relations:
                e1 = rel.get("e1", "").strip()
                e2 = rel.get("e2", "").strip()
                if not (e1 and e2):
                    continue
                m1 = kg_query.find_drug_in_kg(e1)
                m2 = kg_query.find_drug_in_kg(e2)
                if not (m1 and m2):
                    still_pending.append(_make_pending_record(text, e1, e2, label, src))
                    retry_unmatched += 1
                    continue
                valid_rels.append({"e1": e1, "e2": e2})
                retry_matched += 1

            if valid_rels:
                drug_names = []
                seen_drugs = set()
                for rel in valid_rels:
                    for key in ("e1", "e2"):
                        d = rel.get(key, "").strip()
                        if d and d not in seen_drugs:
                            seen_drugs.add(d)
                            drug_names.append(d)

                entry = create_finetune_entry(
                    text=text,
                    drug_names=drug_names,
                    relations=valid_rels,
                    kg_query=kg_query,
                    include_kg_evidence=ablation["include_kg_evidence"],
                    include_gold_label_in_input=ablation["include_gold_label_in_input"],
                    output_mode=ablation["output_mode"],
                    gold_label=label,
                )
                all_entries.append(entry)

            if args.retry_log_every > 0 and i % args.retry_log_every == 0:
                print(
                    f"[Retry] 进度 {i}/{len(grouped_values)} | "
                    f"新增成功 {retry_matched} | 仍未匹配 {retry_unmatched}"
                )

        pending_records = still_pending
        print(
            f"[Retry] 完成: 总处理 {len(retry_items)} | "
            f"新增匹配成功 {retry_matched} | 仍未匹配 {retry_unmatched}"
        )
    else:
        for label, filename in LABEL_CONFIG.items():
            stem = Path(filename).stem
            target_filename = f"{stem}{args.input_suffix}.json"
            input_path = project_root / "data" / target_filename
            if not input_path.exists():
                print(f"[跳过] 文件不存在: {input_path}")
                continue

            if args.use_all_data:
                print(f"\n[{label}] 读取 {input_path.name}，全量模式...")
            else:
                target_count = counts[label]
                print(f"\n[{label}] 读取 {input_path.name}，目标 {target_count} 条...")

            raw_entries = process_label_data(
                input_json=str(input_path),
                output_json=None,
                kg_query=kg_query,
                gold_label=label,
                limit=None,
                verbose=args.verbose,
                include_kg_evidence=ablation["include_kg_evidence"],
                include_gold_label_in_input=ablation["include_gold_label_in_input"],
                output_mode=ablation["output_mode"],
                skip_unmatched=args.skip_unmatched,
                pending_records=pending_records,
            )

            raw_count = len(raw_entries)

            if not args.use_all_data:
                if len(raw_entries) > target_count:
                    raw_entries = random.sample(raw_entries, target_count)
                elif len(raw_entries) < target_count:
                    print(f"  警告: {label} 实际只有 {len(raw_entries)} 条（< 目标 {target_count}）")

                print(f"  [{label}] 原始 {raw_count} 条 -> 采样后 {len(raw_entries)} 条")
            else:
                print(f"  [{label}] 全量保留 {raw_count} 条")

            all_entries.extend(raw_entries)

    random.shuffle(all_entries)

    sidecar_labels: List[Dict[str, Any]] = []
    if args.drop_gold_label:
        cleaned_entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(all_entries):
            copied = dict(entry)
            input_data = dict(copied.get("input", {}))
            gold = input_data.pop("gold_label", None)
            if gold is None:
                # In label-only modes, gold labels are stored in output instead of input.
                gold = copied.get("output")
            copied["input"] = input_data
            cleaned_entries.append(copied)

            sidecar_labels.append({
                "id": idx,
                "gold_label": gold,
                "sentence": input_data.get("sentence", ""),
                "query_group": get_query_group_text(input_data),
            })
        all_entries = cleaned_entries

    if args.output:
        output_path = Path(args.output)
    else:
        if args.input_suffix == "_test":
            default_base = "finetune_dataset_input_test"
        else:
            default_base = "finetune_dataset_input"
        if args.ablation_mode == "full":
            default_name = f"{default_base}.json"
        else:
            default_name = f"{default_base}_{args.ablation_mode}.json"
        output_path = project_root / "data" / default_name

    if args.append_output and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            all_entries = existing + all_entries
            print(f"追加模式: 已有 {len(existing)} 条，本次新增 {len(all_entries) - len(existing)} 条")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # 保存待对齐清单，用于外部名称对齐后 retry
    pending_path = Path(args.pending_output) if args.pending_output else (project_root / "data" / "pending_unmatched_pairs.json")
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    # 去重待对齐记录
    dedup = {}
    for p in pending_records:
        key = (p.get("text", ""), p.get("e1", ""), p.get("e2", ""), p.get("gold_label", ""))
        dedup[key] = p
    pending_records = list(dedup.values())

    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(pending_records, f, ensure_ascii=False, indent=2)
    print(f"待对齐药物对已保存到: {pending_path} (共 {len(pending_records)} 条)")

    after_unmatched = sum(1 for v in kg_query._user_name_map_raw.values() if v is None)
    newly_added_unmatched = max(0, after_unmatched - before_unmatched)
    print(
        f"未匹配药物映射统计: 运行前 {before_unmatched} 条, 运行后 {after_unmatched} 条, "
        f"本次新增 {newly_added_unmatched} 条 (文件: {kg_query.mapping_path})"
    )

    if args.drop_gold_label:
        if args.gold_label_sidecar:
            sidecar_path = Path(args.gold_label_sidecar)
        else:
            default_sidecar = "finetune_dataset_input_test_labels.json" if args.input_suffix == "_test" else "finetune_dataset_input_labels.json"
            sidecar_path = project_root / "data" / default_sidecar
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_labels, f, ensure_ascii=False, indent=2)
        print(f"标签对照已保存到: {sidecar_path}")

    print(f"\n完成！共 {len(all_entries)} 条，已保存到: {output_path}")

    if all_entries:
        print("\n" + "=" * 60)
        print("示例数据（第1条）:")
        print("=" * 60)
        print(json.dumps(all_entries[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

