from data_process.build_prime_kg import load_query_graph
from data_process.paths import PRIMEKG_GRAPH_PKL
import time
import json



class GraphQuery:
    def __init__(self, graph_path: str | None = None):
        # 默认 data/kg/primekg_graph.pkl；允许显式覆盖
        graph_path = graph_path if graph_path is not None else str(PRIMEKG_GRAPH_PKL)
        self.g = load_query_graph(graph_path)
    def neighbors(self, name: str, relation: str | None = None, direction: str = "out"):
        res = []
        for src, dst, rel in self.g.neighbors(name, relation=relation, direction=direction):
            res.append({
                "from": self.g.node_names[src],
                "to": self.g.node_names[dst],
                "relation": rel,
                "from_type": self.g.node_types[src],
                "to_type": self.g.node_types[dst],
                "from_id": src,
                "to_id": dst,
            })
        return res
    def relations_between(self, a: str, b: str):
        return self.g.relations_between(a, b)
    def shortest_path(self, source: str, target: str, max_hops: int = 4, direction: str = "both"):
        steps = self.g.shortest_path_with_edges(source, target, max_hops=max_hops, direction=direction)
        lines = []
        for src, rel, dst in steps:
            lines.append(f"{self.g.node_names[src]} ({self.g.node_types[src]}) --[{rel}]--> {self.g.node_names[dst]} ({self.g.node_types[dst]})")
        return {
            "length": len(steps) + (1 if steps else 0),
            "steps": steps,
            "lines": lines,
            "text": self.g.format_path(steps),
        }
    def _adjacent(self, idx: int, direction: str):
        nbrs = []
        if direction in ("out", "both"):
            nbrs.extend(self.g.out_adj.get(idx, []))
        if direction in ("in", "both"):
            nbrs.extend([(n, r) for n, r in self.g.in_adj.get(idx, [])])
        return nbrs
    def simple_paths_generator(self, source: str, target: str, cutoff: int = 6, direction: str = "both"):
        starts = self.g.name_to_indices.get(source, [])
        targets = set(self.g.name_to_indices.get(target, []))
        if not starts or not targets:
            return
        stack = []
        for s in starts:
            stack.append((s, [s], set([s])))
        while stack:
            cur, path, seen = stack.pop()
            if len(path) > cutoff:
                continue
            if cur in targets and len(path) > 1:
                yield list(path)
                continue
            for n, _r in self._adjacent(cur, direction):
                if n in seen:
                    continue
                new_seen = set(seen)
                new_seen.add(n)
                stack.append((n, path + [n], new_seen))
    def find_entity_paths(self, source: str, target: str, max_paths: int = 3, cutoff: int = 6, direction: str = "both"):
        gen = self.simple_paths_generator(source, target, cutoff=cutoff, direction=direction)
        paths = []
        taken = 0
        for p in gen:
            info = {"length": len(p), "nodes": [], "relations": []}
            for i in range(len(p) - 1):
                u = p[i]
                v = p[i + 1]
                u_name = self.g.node_names[u]
                v_name = self.g.node_names[v]
                rel = None
                for n, r in self.g.out_adj.get(u, []):
                    if n == v:
                        rel = self.g.relation_types[r]
                        break
                if rel is None:
                    for n, r in self.g.in_adj.get(v, []):
                        if n == u:
                            rel = self.g.relation_types[r]
                            break
                info["nodes"].append({"id": u, "name": u_name, "type": self.g.node_types[u]})
                info["relations"].append({"from": u_name, "to": v_name, "relation": rel or "unknown", "from_type": self.g.node_types[u], "to_type": self.g.node_types[v]})
            last = p[-1]
            info["nodes"].append({"id": last, "name": self.g.node_names[last], "type": self.g.node_types[last]})
            paths.append(info)
            taken += 1
            if taken >= max_paths:
                break
        return paths

    def simple_paths_generator_with_progress(self, source: str, target: str, cutoff: int = 6, direction: str = "both", report_interval: int = 50000, time_interval_sec: float = 2.0):
        starts = self.g.name_to_indices.get(source, [])
        targets = set(self.g.name_to_indices.get(target, []))
        if not starts or not targets:
            return
        stack = []
        for s in starts:
            stack.append((s, [s], set([s])))
        expansions = 0
        last_report_exp = 0
        last_report_time = time.time()
        while stack:
            cur, path, seen = stack.pop()
            if len(path) > cutoff:
                continue
            if cur in targets and len(path) > 1:
                yield {"kind": "path", "path": list(path)}
                continue
            for n, _r in self._adjacent(cur, direction):
                if n in seen:
                    continue
                expansions += 1
                if expansions - last_report_exp >= report_interval or (time.time() - last_report_time) >= time_interval_sec:
                    last_report_exp = expansions
                    last_report_time = time.time()
                    yield {"kind": "progress", "expansions": expansions, "stack": len(stack), "depth": len(path)}
                new_seen = set(seen)
                new_seen.add(n)
                stack.append((n, path + [n], new_seen))

    def build_nodes_edges_from_steps(self, steps):
        node_set = set()
        edges = []
        for src, rel, dst in steps:
            node_set.add(src)
            node_set.add(dst)
            edges.append({"from": src, "to": dst, "label": rel})
        nodes = []
        for idx in sorted(node_set):
            nodes.append({"id": idx, "label": self.g.node_names[idx], "group": self.g.node_types[idx]})
        return nodes, edges

    def build_neighbors_subgraph(self, name: str, k_out: int = 25, k_in: int = 25):
        out_list = self.neighbors(name, direction="out")[:k_out]
        in_list = self.neighbors(name, direction="in")[:k_in]
        node_set = set()
        edges = []
        for item in out_list:
            node_set.add(item["from_id"])
            node_set.add(item["to_id"])
            edges.append({"from": item["from_id"], "to": item["to_id"], "label": item["relation"]})
        for item in in_list:
            node_set.add(item["from_id"])
            node_set.add(item["to_id"])
            edges.append({"from": item["from_id"], "to": item["to_id"], "label": item["relation"]})
        nodes = []
        for idx in sorted(node_set):
            nodes.append({"id": idx, "label": self.g.node_names[idx], "group": self.g.node_types[idx]})
        return nodes, edges

    def export_visjs_html(self, nodes, edges, output_path: str):
        html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <script src=\"https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js\"></script>
  <style>#mynetwork{width:100%;height:800px;border:1px solid #ddd;}</style>
</head>
<body>
<div id=\"mynetwork\"></div>
<script>
const nodes = new vis.DataSet(%NODES%);
const edges = new vis.DataSet(%EDGES%);
const container = document.getElementById('mynetwork');
const data = { nodes, edges };
const options = {
  nodes: { shape: 'dot', size: 12 },
  edges: { arrows: { to: { enabled: true } }, font: { size: 10, align: 'middle' } },
  physics: { stabilization: true }
};
new vis.Network(container, data, options);
</script>
</body>
</html>
"""
        html = html.replace("%NODES%", json.dumps(nodes, ensure_ascii=False))
        html = html.replace("%EDGES%", json.dumps(edges, ensure_ascii=False))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

def format_relations_lines(path_info):
    lines = []
    for rel in path_info.get("relations", []):
        lines.append(f"{rel['from']} ({rel['from_type']}) --[{rel['relation']}]--> {rel['to']} ({rel['to_type']})")
    return lines