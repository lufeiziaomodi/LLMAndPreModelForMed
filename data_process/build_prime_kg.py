import pandas as pd
import torch
import numpy as np
import pickle
import sys
from pathlib import Path

# 允许直接 `python data_process/build_prime_kg.py` 运行
_HERE = Path(__file__).resolve()
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from data_process.paths import KG_CSV, PRIMEKG_GRAPH_PKL, PRIMEKG_PYG_PT, ensure_dir, DATA_KG

try:
    from torch_geometric.data import Data
except Exception:
    Data = None


def _build_core_graph(df):
    """
    根据 CSV 构建基础图结构，同时保留关系的通用与展示名称，便于查询与可视化。
    """
    # 兼容列名：部分数据列写作 generelation
    general_rel_col = None
    for cand in ("general_relation", "generelation"):
        if cand in df.columns:
            general_rel_col = cand
            break
    display_rel_col = "display_relation" if "display_relation" in df.columns else None
    rel_col = "relation" if "relation" in df.columns else None

    node_ids = {}
    node_types = []
    node_names = []
    all_nodes = set()
    for _, row in df.iterrows():
        all_nodes.add((row["x_index"], row["x_name"], row["x_type"]))
        all_nodes.add((row["y_index"], row["y_name"], row["y_type"]))
    for idx, (node_id, name, node_type) in enumerate(all_nodes):
        node_ids[node_id] = idx
        node_names.append(name)
        node_types.append(node_type)

    # 构建关系信息（原始 / 通用 / 展示）
    relation_tuples = set()
    for _, row in df.iterrows():
        relation_tuples.add(
            (
                row[rel_col] if rel_col else "",
                row[general_rel_col] if general_rel_col else "",
                row[display_rel_col] if display_rel_col else "",
            )
        )
    relation_infos = []
    relation_to_idx = {}
    for idx, (rel, general_rel, display_rel) in enumerate(sorted(relation_tuples)):
        relation_infos.append(
            {
                "relation": rel,
                "general_relation": general_rel,
                "display_relation": display_rel,
            }
        )
        relation_to_idx[(rel, general_rel, display_rel)] = idx

    edge_index_list = []
    edge_attr_list = []
    for _, row in df.iterrows():
        src = node_ids[row["x_index"]]
        dst = node_ids[row["y_index"]]
        key = (
            row[rel_col] if rel_col else "",
            row[general_rel_col] if general_rel_col else "",
            row[display_rel_col] if display_rel_col else "",
        )
        relation_idx = relation_to_idx[key]
        edge_index_list.append([src, dst])
        edge_attr_list.append(relation_idx)
    # 为兼容旧接口，提供 relation_types（使用原始 relation 字段）
    relation_types = [info["relation"] for info in relation_infos]
    return (
        node_ids,
        node_names,
        node_types,
        relation_infos,
        relation_types,
        edge_index_list,
        edge_attr_list,
    )

def build_primekg_pyg_graph(csv_path, output_path=str(PRIMEKG_PYG_PT)):
    df = pd.read_csv(csv_path, low_memory=False)
    (
        node_ids,
        node_names,
        node_types,
        relation_infos,
        relation_types,
        edge_index_list,
        edge_attr_list,
    ) = _build_core_graph(df)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    x = torch.ones(len(node_ids), 1)
    if Data is None:
        return None
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_names=node_names,
        node_types=node_types,
        relation_types=relation_types,
        relation_infos=relation_infos,
        relation_general=[info["general_relation"] for info in relation_infos],
        relation_display=[info["display_relation"] for info in relation_infos],
    )
    id_to_idx = {int(k): int(v) for k, v in node_ids.items()}
    name_to_indices = {}
    for idx, name in enumerate(node_names):
        name_to_indices.setdefault(name, []).append(idx)
    data.id_to_idx = id_to_idx
    data.name_to_indices = name_to_indices
    torch.save(data, output_path)
    return data

class QueryGraph:
    def __init__(self, node_names, node_types, relation_infos, edge_index_list, edge_attr_list, id_to_idx):
        self.node_names = node_names
        self.node_types = node_types
        self.relation_infos = relation_infos
        self.relation_types = [info["relation"] for info in relation_infos]
        self.relation_general = [info["general_relation"] for info in relation_infos]
        self.relation_display = [info["display_relation"] for info in relation_infos]
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list
        self.id_to_idx = id_to_idx
        # 预构建索引与邻接，提升查询速度
        self.name_to_indices = {}
        for idx, name in enumerate(node_names):
            self.name_to_indices.setdefault(name, []).append(idx)
        self.out_adj = {}
        self.in_adj = {}
        for i, (src, dst) in enumerate(edge_index_list):
            rel = edge_attr_list[i]
            self.out_adj.setdefault(src, []).append((dst, rel))
            self.in_adj.setdefault(dst, []).append((src, rel))

    def _relation_label(self, rel_idx, prefer_display=True):
        if prefer_display and self.relation_display:
            label = self.relation_display[rel_idx]
            if label:
                return label
        return self.relation_types[rel_idx]

    def neighbors(self, node, relation=None, direction="out", use_display=True):
        """
        获取邻居。
        relation 可传原始 relation 或 general_relation / display_relation。
        """
        if isinstance(node, str):
            indices = self.name_to_indices.get(node, [])
        else:
            indices = [node]

        def _match(rel_idx, target):
            if target is None:
                return True
            return (
                self.relation_types[rel_idx] == target
                or self.relation_general[rel_idx] == target
                or self.relation_display[rel_idx] == target
            )

        result = []
        for idx in indices:
            if direction in ("out", "both"):
                for n, r in self.out_adj.get(idx, []):
                    if _match(r, relation):
                        result.append((idx, n, self._relation_label(r, use_display)))
            if direction in ("in", "both"):
                for n, r in self.in_adj.get(idx, []):
                    if _match(r, relation):
                        result.append((n, idx, self._relation_label(r, use_display)))
        return result

    def relations_between(self, a, b):
        if isinstance(a, str):
            a_indices = self.name_to_indices.get(a, [])
        else:
            a_indices = [a]
        if isinstance(b, str):
            b_indices = self.name_to_indices.get(b, [])
        else:
            b_indices = [b]
        rels = []
        for ai in a_indices:
            for bi in b_indices:
                for n, r in self.out_adj.get(ai, []):
                    if n == bi:
                        rels.append(self.relation_types[r])
                for n, r in self.in_adj.get(ai, []):
                    if n == bi:
                        rels.append(self.relation_types[r])
        return list(set(rels))

    def shortest_path(self, a, b, max_hops=4, direction="out", use_display=True):
        if isinstance(a, str):
            starts = self.name_to_indices.get(a, [])
        else:
            starts = [a]
        if isinstance(b, str):
            targets = set(self.name_to_indices.get(b, []))
        else:
            targets = {b}
        from collections import deque

        visited = set()
        for s in starts:
            dq = deque([(s, [])])
            visited.add(s)
            hops = {s: 0}
            while dq:
                cur, path = dq.popleft()
                if cur in targets:
                    return path + [(cur, None)]
                if hops.get(cur, 0) >= max_hops:
                    continue
                neighbors = []
                if direction in ("out", "both"):
                    neighbors.extend([(n, r) for n, r in self.out_adj.get(cur, [])])
                if direction in ("in", "both"):
                    neighbors.extend([(n, r) for n, r in self.in_adj.get(cur, [])])
                for n, r in neighbors:
                    if n not in visited:
                        visited.add(n)
                        hops[n] = hops.get(cur, 0) + 1
                        dq.append((n, path + [(cur, self._relation_label(r, use_display))]))
        return []

    def nodes_by_type(self, t):
        return [i for i, tp in enumerate(self.node_types) if tp == t]

    def shortest_path_with_edges(self, a, b, max_hops=4, direction="out", use_display=True):
        if isinstance(a, str):
            starts = self.name_to_indices.get(a, [])
        else:
            starts = [a]
        if isinstance(b, str):
            targets = set(self.name_to_indices.get(b, []))
        else:
            targets = {b}
        from collections import deque

        dq = deque()
        prev = {}
        depth = {}
        for s in starts:
            dq.append(s)
            prev[s] = None
            depth[s] = 0
        visited = set(starts)
        found = None
        while dq:
            cur = dq.popleft()
            if cur in targets:
                found = cur
                break
            if depth.get(cur, 0) >= max_hops:
                continue
            neighbors = []
            if direction in ("out", "both"):
                neighbors.extend(self.out_adj.get(cur, []))
            if direction in ("in", "both"):
                neighbors.extend(self.in_adj.get(cur, []))
            for n, r in neighbors:
                if n not in visited:
                    visited.add(n)
                    prev[n] = (cur, r)
                    depth[n] = depth.get(cur, 0) + 1
                    dq.append(n)
        if found is None:
            return []
        steps = []
        cur = found
        while prev.get(cur) is not None:
            p, r_idx = prev[cur]
            steps.append((p, self._relation_label(r_idx, use_display), cur))
            cur = p
        steps.reverse()
        return steps

    def k_simple_paths(self, a, b, k=5, max_hops=5, direction="out", use_display=True):
        """
        返回长度不一的前 k 条简单路径（按 hop 从短到长），用于挖掘多样链路。
        """
        if isinstance(a, str):
            starts = self.name_to_indices.get(a, [])
        else:
            starts = [a]
        if isinstance(b, str):
            targets = set(self.name_to_indices.get(b, []))
        else:
            targets = {b}
        from collections import deque

        results = []
        for s in starts:
            dq = deque()
            dq.append((s, []))
            while dq and len(results) < k:
                cur, path = dq.popleft()
                if len(path) > max_hops:
                    continue
                if path and cur in targets:
                    results.append(path)
                    # 继续探索，确保获得不同长度的路径
                neighbors = []
                if direction in ("out", "both"):
                    neighbors.extend(self.out_adj.get(cur, []))
                if direction in ("in", "both"):
                    neighbors.extend(self.in_adj.get(cur, []))
                for n, r in neighbors:
                    # 保证简单路径，避免环
                    visited_nodes = [step[0] for step in path] + [cur]
                    if n in visited_nodes:
                        continue
                    dq.append(
                        (
                            n,
                            path
                            + [
                                (
                                    cur,
                                    self._relation_label(r, use_display),
                                    n,
                                )
                            ],
                        )
                    )
        return results[:k]

    def format_path(self, steps):
        if not steps:
            return ""
        parts = []
        for src, rel, dst in steps:
            parts.append(f"{self.node_names[src]} -[{rel}]-> {self.node_names[dst]}")
        return " -> ".join(parts)

def build_query_graph(csv_path, output_path=str(PRIMEKG_GRAPH_PKL)):
    df = pd.read_csv(csv_path, low_memory=False)
    (
        node_ids,
        node_names,
        node_types,
        relation_infos,
        relation_types,
        edge_index_list,
        edge_attr_list,
    ) = _build_core_graph(df)
    id_to_idx = {int(k): int(v) for k, v in node_ids.items()}
    payload = {
        "node_names": node_names,
        "node_types": node_types,
        "relation_infos": relation_infos,
        "relation_types": relation_types,
        "relation_display": [info["display_relation"] for info in relation_infos],
        "relation_general": [info["general_relation"] for info in relation_infos],
        "edge_index_list": edge_index_list,
        "edge_attr_list": edge_attr_list,
        "id_to_idx": id_to_idx,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    return QueryGraph(
        node_names,
        node_types,
        relation_infos,
        edge_index_list,
        edge_attr_list,
        id_to_idx,
    )

def load_query_graph(path=str(PRIMEKG_GRAPH_PKL)):
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        csv_path = str(KG_CSV)
        df = pd.read_csv(csv_path, low_memory=False)
        (
            node_ids,
            node_names,
            node_types,
            relation_infos,
            relation_types,
            edge_index_list,
            edge_attr_list,
        ) = _build_core_graph(df)
        id_to_idx = {int(k): int(v) for k, v in node_ids.items()}
        payload = {
            "node_names": node_names,
            "node_types": node_types,
            "relation_infos": relation_infos,
            "relation_types": relation_types,
            "relation_display": [info["display_relation"] for info in relation_infos],
            "relation_general": [info["general_relation"] for info in relation_infos],
            "edge_index_list": edge_index_list,
            "edge_attr_list": edge_attr_list,
            "id_to_idx": id_to_idx,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    if isinstance(payload, dict):
        # 兼容旧版本：若无 relation_infos，则从 relation_types 构造占位字段
        if "relation_infos" not in payload:
            payload["relation_infos"] = [
                {
                    "relation": rel,
                    "general_relation": "",
                    "display_relation": "",
                }
                for rel in payload.get("relation_types", [])
            ]
        return QueryGraph(
            payload["node_names"],
            payload["node_types"],
            payload["relation_infos"],
            payload["edge_index_list"],
            payload["edge_attr_list"],
            payload["id_to_idx"],
        )
    try:
        return QueryGraph(
            payload.node_names,
            payload.node_types,
            payload.relation_infos,
            payload.edge_index_list,
            payload.edge_attr_list,
            payload.id_to_idx,
        )
    except Exception:
        csv_path = str(KG_CSV)
        return build_query_graph(csv_path, output_path=path)

 


if __name__ == "__main__":
    # 统一从 paths.py 常量取路径；输入 data/kg/kg.csv，输出 data/kg/primekg_*
    ensure_dir(DATA_KG)
    csv_path = str(KG_CSV)
    g = build_query_graph(csv_path, output_path=str(PRIMEKG_GRAPH_PKL))
    if Data is not None:
        _ = build_primekg_pyg_graph(csv_path, output_path=str(PRIMEKG_PYG_PT))
    print("构建完成")
    print(f"节点数: {len(g.node_names)}")
    print(f"边数: {len(g.edge_index_list)}")
    if g.node_names:
        sample_name = g.node_names[0]
        nb = g.neighbors(sample_name, direction="out")
        print(f"示例节点: {sample_name}")
        print(f"邻居数: {len(nb)}")
