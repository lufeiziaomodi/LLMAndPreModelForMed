"""
基于实体名查询有效路径的便捷工具。

功能：
- 按实体名查找多条简单路径，支持长度下限（例如 3）与上限 k。
- 路径可按展示名输出，便于阅读。
- 从源节点构建树形结构，target_types 作为叶子节点。
"""

from typing import List, Dict, Any, Set, Optional
import json

from data_process.build_prime_kg import load_query_graph


class PathQuery:
    def __init__(self, graph_path: str = "../data/primekg_graph.pkl"):
        self.g = load_query_graph(graph_path)
        self._name_to_indices_ci = {}
        for name, inds in self.g.name_to_indices.items():
            key = name.casefold()
            self._name_to_indices_ci.setdefault(key, []).extend(inds)

    def get_path_groups(self, paths: List[Dict[str, Any]], prefix_len: int | None = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取路径分组信息，返回格式化的分组结果。
        返回: {prefix_text: [paths]}，其中 prefix_text 是路径前缀的文本表示
        """
        groups = self._group_paths_by_prefix(paths, prefix_len=prefix_len)
        formatted = {}
        for prefix, group_paths in groups.items():
            if not group_paths:
                continue
            # 使用第一条路径的前缀部分作为文本表示
            first_path = group_paths[0]
            steps = first_path["steps"]
            if prefix_len is None:
                prefix_steps = steps[:-1]
            else:
                prefix_steps = steps[:prefix_len]
            prefix_text = self.g.format_path(prefix_steps) if prefix_steps else "root"
            formatted[prefix_text] = group_paths
        return formatted

    def _iter_paths_backtracking(
        self,
        sources: List[int],
        is_target,
        min_hops: int,
        max_hops: int,
        max_results: int,
        direction: str,
        use_display: bool,
        max_expansions: int | None = None,
        branch_limit: int | None = None,
        progress_cb=None,
        progress_every: int = 20000,
        progress_secs: float = 2.0,
        diversity_aware: bool = False,
    ):
        """
        内存友好的简单路径枚举（迭代加回溯），避免对 visited 反复复制。
        - branch_limit: 每个节点扩展的邻居上限，用于控制爆炸。
        - max_expansions: 扩展上限，防止长时间/大内存。
        - diversity_aware: 是否启用多样性感知（记录已探索的路径前缀，优先探索新分支）
        """
        import time
        import random

        results = []
        visited = set()
        stack = []  # (node, iter(neighbors), path_edges)
        expansions = 0
        last_report_exp = 0
        last_report_time = time.time()
        
        # 多样性感知：记录已探索的路径前缀模式
        explored_prefixes = set() if diversity_aware else None

        def iter_neighbors(node):
            neighbors = []
            if direction in ("out", "both"):
                neighbors.extend(self.g.out_adj.get(node, []))
            if direction in ("in", "both"):
                neighbors.extend(self.g.in_adj.get(node, []))
            if branch_limit is not None and len(neighbors) > branch_limit:
                # 多样性感知：如果启用，随机打乱以增加探索多样性
                if diversity_aware:
                    random.shuffle(neighbors)
                neighbors = neighbors[:branch_limit]
            return iter(neighbors)

        def maybe_report(depth_len):
            nonlocal last_report_exp, last_report_time
            now = time.time()
            if not progress_cb:
                return
            if (expansions - last_report_exp >= progress_every) or (now - last_report_time >= progress_secs):
                progress_cb(
                    {
                        "expansions": expansions,
                        "stack": len(stack),
                        "found": len(results),
                        "depth": depth_len,
                    }
                )
                last_report_exp = expansions
                last_report_time = now

        for s in sources:
            visited.clear()
            visited.add(s)
            stack = [(s, iter_neighbors(s), [])]
            while stack and len(results) < max_results:
                node, nbr_iter, edges = stack[-1]
                depth_len = len(edges)
                # 命中目标
                if depth_len >= min_hops and is_target(node) and edges:
                    # 多样性感知：记录路径前缀，避免重复探索相同模式
                    if diversity_aware and explored_prefixes is not None:
                        # 记录去掉最后一步的前缀
                        if len(edges) > 1:
                            prefix = tuple((src, rel) for src, rel, _ in edges[:-1])
                            if prefix in explored_prefixes:
                                # 跳过相同前缀的路径（可选：也可以保留但降低优先级）
                                pass
                            explored_prefixes.add(prefix)
                    results.append(edges)
                    maybe_report(depth_len)
                    # 不 return，继续找其他路径

                if depth_len >= max_hops:
                    stack.pop()
                    visited.remove(node)
                    continue

                try:
                    nxt, rel_idx = next(nbr_iter)
                except StopIteration:
                    stack.pop()
                    visited.remove(node)
                    continue

                if nxt in visited:
                    continue

                if max_expansions is not None and expansions >= max_expansions:
                    maybe_report(depth_len)
                    return results

                label = self.g.relation_display[rel_idx] if use_display else self.g.relation_types[rel_idx]
                visited.add(nxt)
                stack.append((nxt, iter_neighbors(nxt), edges + [(node, label, nxt)]))
                expansions += 1
                maybe_report(len(edges) + 1)

        return results

    def find_paths_by_name(
        self,
        source: str,
        target: str,
        min_hops: int = 3,
        max_hops: int = 6,
        k: int = 50,
        direction: str = "both",
        use_display: bool = True,
        max_expansions: int | None = 1_000_000,
        branch_limit: int | None = 200,
        progress_cb=None,
    ) -> List[Dict[str, Any]]:
        """
        查找从 source 到 target 的简单路径，支持进度回调与长度范围。
        - k: 返回的最大路径条数（默认 50，避免无限制搜索占用内存）
        - branch_limit: 每节点扩展邻居上限，防止高出度节点导致爆炸
        - max_expansions: 搜索扩展上限，防止长时间/高内存
        progress_cb: lambda info: ...，info={"expansions","stack","found","depth"}
        """
        starts = self._name_to_indices_ci.get(source.casefold(), [])
        targets = set(self._name_to_indices_ci.get(target.casefold(), []))
        if not starts or not targets:
            return []

        def is_target(idx: int):
            return idx in targets

        raw_paths = self._iter_paths_backtracking(
            starts,
            is_target,
            min_hops=min_hops,
            max_hops=max_hops,
            max_results=k,
            direction=direction,
            use_display=use_display,
            max_expansions=max_expansions,
            branch_limit=branch_limit,
            progress_cb=progress_cb,
        )

        results = []
        for steps in raw_paths:
            hop = len(steps)
            text = self.g.format_path(steps)
            lines = [
                f"{self.g.node_names[src]} ({self.g.node_types[src]}) --[{rel}]--> {self.g.node_names[dst]} ({self.g.node_types[dst]})"
                for src, rel, dst in steps
            ]
            results.append({"length": hop, "steps": steps, "text": text, "lines": lines})
        return results

    def _group_paths_by_prefix(self, paths: List[Dict[str, Any]], prefix_len: int | None = None) -> Dict[tuple, List[Dict[str, Any]]]:
        """
        按路径前缀分组路径。相同前缀的路径（仅最后一步不同）会被归为一组。
        - prefix_len: 前缀长度（边数），None 表示使用 len(steps)-1（去掉最后一步）
        """
        groups = {}
        for path in paths:
            steps = path["steps"]
            if not steps:
                continue
            if prefix_len is None:
                # 默认去掉最后一步作为前缀
                prefix = tuple((src, rel) for src, rel, _ in steps[:-1])
            else:
                prefix = tuple((src, rel) for src, rel, _ in steps[:prefix_len])
            groups.setdefault(prefix, []).append(path)
        return groups

    def _diversify_paths(
        self,
        paths: List[Dict[str, Any]],
        k: int,
        max_per_group: int = 2,
        prefix_len: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        路径多样性过滤：按前缀分组，每组最多保留 max_per_group 条路径，确保返回的 k 条路径来自不同的路径模式。
        - max_per_group: 每组最多保留的路径数（默认 2，即同一前缀最多保留 2 条不同终点的路径）
        - prefix_len: 前缀长度，None 表示使用 len(steps)-1
        """
        if not paths:
            return []
        
        groups = self._group_paths_by_prefix(paths, prefix_len=prefix_len)
        
        # 按组大小排序，优先保留路径数多的组（但每组最多取 max_per_group 条）
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        diversified = []
        for prefix, group_paths in sorted_groups:
            # 每组最多保留 max_per_group 条
            taken = min(max_per_group, len(group_paths))
            diversified.extend(group_paths[:taken])
            if len(diversified) >= k:
                break
        
        return diversified[:k]

    def explore_from_source(
        self,
        source: str,
        target_names: List[str] | None = None,
        target_types: List[str] | None = None,
        min_hops: int = 2,
        max_hops: int = 4,
        k: int = 50,
        direction: str = "both",
        use_display: bool = True,
        max_expansions: int | None = 1_000_000,
        branch_limit: int | None = 200,
        progress_cb=None,
        diversify: bool = True,
        max_per_group: int = 2,
        search_more: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        从单个源实体出发探索多样路径：
        - target_names: 目标实体名集合（可选）
        - target_types: 目标实体类型集合（可选，如 ["Disease","Drug"]）
        - 若两者皆为空，则返回前 k 条达到任意节点的路径（长度过滤后）
        - diversify: 是否启用路径多样性过滤（默认 True），将相同前缀的路径归类
        - max_per_group: 每组（相同前缀）最多保留的路径数（默认 2）
        - search_more: 多样性模式下，搜索更多路径后再过滤（默认 3 倍，即搜索 k*3 条）
        """
        starts = self._name_to_indices_ci.get(source.casefold(), [])
        if not starts:
            return []

        target_set = set()
        if target_names:
            for name in target_names:
                target_set.update(self._name_to_indices_ci.get(name.casefold(), []))

        def is_target(idx: int):
            if target_set and idx in target_set:
                return True
            if target_types and self.g.node_types[idx] in target_types:
                return True
            return not target_set and not target_types  # 若未指定目标，则任意节点皆可作为终点

        # 如果启用多样性，搜索更多路径以便过滤
        search_k = k * search_more if diversify else k
        
        raw_paths = self._iter_paths_backtracking(
            starts,
            is_target,
            min_hops=min_hops,
            max_hops=max_hops,
            max_results=search_k,
            direction=direction,
            use_display=use_display,
            max_expansions=max_expansions,
            branch_limit=branch_limit,
            progress_cb=progress_cb,
            diversity_aware=diversify,  # 启用多样性感知搜索
        )

        results = []
        for steps in raw_paths:
            hop = len(steps)
            text = self.g.format_path(steps)
            lines = [
                f"{self.g.node_names[src]} ({self.g.node_types[src]}) --[{rel}]--> {self.g.node_names[dst]} ({self.g.node_types[dst]})"
                for src, rel, dst in steps
            ]
            results.append({"length": hop, "steps": steps, "text": text, "lines": lines})
        
        # 应用多样性过滤
        if diversify and results:
            results = self._diversify_paths(results, k=k, max_per_group=max_per_group)
        
        return results

    def build_tree_from_source(
        self,
        source: str,
        target_types: List[str],
        max_depth: int = 4,
        direction: str = "both",
        use_display: bool = True,
        branch_limit: int | None = 200,
        max_expansions: int | None = 1_000_000,
        progress_cb=None,
        allow_cycles: bool = False,
    ) -> Dict[str, Any]:
        """
        从源节点构建树形结构，target_types 作为叶子节点。
        
        优化特性：
        1. 叶子节点限制：只有target_types指定的类型才能成为叶子节点
        2. 节点合并：同一父节点下，相同关系和类型的子节点会合并name为数组，减少重复
        3. 关系多样性：重点关注边的多样性，每个不同的关系都会保留
        
        - source: 源实体名称
        - target_types: 目标实体类型列表（如 ["Drug"]），这些类型的节点将作为叶子节点
        - max_depth: 树的最大深度（从源节点开始，深度从0开始）
        - direction: "out" | "in" | "both"
        - use_display: 是否使用关系展示名
        - branch_limit: 每个节点扩展的邻居上限
        - max_expansions: 最大扩展次数限制
        - progress_cb: 进度回调函数
        - allow_cycles: 是否允许循环（默认False，避免重复节点）
        
        返回树形JSON结构：
        {
            "root": {
                "id": node_idx,
                "name": "entity_name" | ["name1", "name2", ...],  # 单个节点为字符串，合并节点为数组
                "type": "entity_type",
                "depth": 0,
                "is_leaf": false,  # 只有target_types指定的类型才为true
                "relation": "relation_label",  # 子节点才有此字段
                "count": 2,  # 合并节点才有此字段，表示合并的节点数量
                "children": [...]
            },
            "metadata": {
                "source": "source_name",
                "target_types": ["Drug"],
                "max_depth": 4,
                "total_nodes": 100,
                "total_edges": 150,
                "leaf_count": 20,
                "relation_diversity": 15,  # 关系多样性：唯一关系的数量
                "unique_relations": ["relation1", "relation2", ...]  # 所有唯一关系列表
            }
        }
        """
        from collections import deque
        
        starts = self._name_to_indices_ci.get(source.casefold(), [])
        if not starts:
            return {"root": None, "metadata": {"error": "Source not found"}}
        
        # 使用第一个匹配的源节点
        root_idx = starts[0]
        root_type = self.g.node_types[root_idx]
        root_name = self.g.node_names[root_idx]
        
        # 处理大小写不敏感：将target_types转换为小写集合，同时保留原始值用于显示
        if target_types:
            # 扩展 PrimeKG 的特殊类型 gene/protein
            expanded_types = set(target_types)
            for t in target_types:
                if t.lower() in ("gene", "protein"):
                    expanded_types.add("gene/protein")
                    expanded_types.add("Gene/Protein")
            target_types = list(expanded_types)

        target_types_lower = {t.lower() for t in target_types} if target_types else set()
        target_types_set = set(target_types) if target_types else set()
        
        # 调试信息：检查根节点的邻居
        out_neighbors = len(self.g.out_adj.get(root_idx, []))
        in_neighbors = len(self.g.in_adj.get(root_idx, []))
        debug_info = {
            "root_name": root_name,
            "root_type": root_type,
            "root_id": root_idx,
            "out_neighbors": out_neighbors,
            "in_neighbors": in_neighbors,
            "target_types_provided": target_types,
            "target_types_lower": list(target_types_lower),
        }
        
        # 树节点结构：{node_idx: {"id", "name", "type", "depth", "is_leaf", "children": []}}
        tree_nodes = {}
        visited = set() if not allow_cycles else None
        expansions = 0
        last_report_exp = 0
        import time
        last_report_time = time.time()
        
        def maybe_report(depth_val, queue_size, found_nodes):
            nonlocal last_report_exp, last_report_time
            now = time.time()
            if not progress_cb:
                return
            if (expansions - last_report_exp >= 20000) or (now - last_report_time >= 2.0):
                progress_cb({
                    "expansions": expansions,
                    "queue": queue_size,
                    "found": found_nodes,
                    "depth": depth_val,
                })
                last_report_exp = expansions
                last_report_time = now
        
        # BFS构建树：记录父子关系和边信息
        # parent_children: {parent_idx: [(child_idx, relation_label)]}
        parent_children = {}
        node_info = {}  # {node_idx: {"id", "name", "type", "depth", "is_leaf"}}
        
        queue = deque()
        queue.append((root_idx, 0))  # (node_idx, depth)
        
        if visited is not None:
            visited.add(root_idx)
        
        # 初始化根节点
        root_type = self.g.node_types[root_idx]
        node_info[root_idx] = {
            "id": root_idx,
            "name": self.g.node_names[root_idx],
            "type": root_type,
            "depth": 0,
            "is_leaf": False,
        }
        parent_children[root_idx] = []
        
        while queue:
            node_idx, depth = queue.popleft()
            
            # 检查是否达到最大深度
            # 优化1: 只有目标类型才能成为叶子节点，达到最大深度但不是目标类型的节点不处理
            if depth >= max_depth:
                node_type = self.g.node_types[node_idx]
                node_type_lower = node_type.lower() if node_type else ""
                is_target_type = (node_type in target_types_set) or (node_type_lower in target_types_lower)
                if is_target_type and node_idx in node_info:
                    node_info[node_idx]["is_leaf"] = True
                continue
            
            # 检查是否是目标类型（叶子节点）
            # 使用大小写不敏感比较
            node_type = self.g.node_types[node_idx]
            node_type_lower = node_type.lower() if node_type else ""
            is_target_type = (node_type in target_types_set) or (node_type_lower in target_types_lower)
            
            # 关键修复：根节点（depth=0）应该始终扩展，即使它是目标类型
            # 只有非根节点的目标类型才跳过扩展
            if is_target_type and depth > 0:
                if node_idx in node_info:
                    node_info[node_idx]["is_leaf"] = True
                continue
            
            # 获取邻居节点
            neighbors = []
            if direction in ("out", "both"):
                neighbors.extend(self.g.out_adj.get(node_idx, []))
            if direction in ("in", "both"):
                neighbors.extend(self.g.in_adj.get(node_idx, []))
            
            # 限制分支数
            if branch_limit is not None and len(neighbors) > branch_limit:
                neighbors = neighbors[:branch_limit]
            
            # 处理邻居
            for nbr_idx, rel_idx in neighbors:
                # 检查扩展限制
                if max_expansions is not None and expansions >= max_expansions:
                    maybe_report(depth, len(queue), len(node_info))
                    break
                
                # 检查是否已访问（避免循环）
                if visited is not None:
                    if nbr_idx in visited:
                        continue
                    visited.add(nbr_idx)
                
                # 检查是否达到最大深度
                if depth + 1 > max_depth:
                    continue
                
                # 检查邻居是否是目标类型（大小写不敏感）
                nbr_type = self.g.node_types[nbr_idx]
                nbr_type_lower = nbr_type.lower() if nbr_type else ""
                is_nbr_target = (nbr_type in target_types_set) or (nbr_type_lower in target_types_lower)
                
                # 优化1: 只有目标类型才能成为叶子节点
                # 如果达到最大深度但不是目标类型，不添加为叶子节点，而是跳过
                if depth + 1 >= max_depth and not is_nbr_target:
                    continue  # 跳过非目标类型且达到最大深度的节点
                
                # 创建子节点信息
                if nbr_idx not in node_info:
                    node_info[nbr_idx] = {
                        "id": nbr_idx,
                        "name": self.g.node_names[nbr_idx],
                        "type": nbr_type,
                        "depth": depth + 1,
                        "is_leaf": is_nbr_target,  # 只有目标类型才是叶子节点
                    }
                    parent_children[nbr_idx] = []
                
                # 添加关系（重点关注关系的多样性）
                rel_label = self.g.relation_display[rel_idx] if use_display else self.g.relation_types[rel_idx]
                if node_idx not in parent_children:
                    parent_children[node_idx] = []
                parent_children[node_idx].append((nbr_idx, rel_label))
                
                # 如果不是叶子节点（即不是目标类型），加入队列继续扩展
                if not is_nbr_target and depth + 1 < max_depth:
                    queue.append((nbr_idx, depth + 1))
                
                expansions += 1
                maybe_report(depth + 1, len(queue), len(node_info))
        
        # 构建递归树结构（优化版本：合并同层同类型节点，重点关注关系多样性）
        build_visited = set()  # 防止递归时出现循环
        
        def build_node(node_idx: int) -> Dict:
            """递归构建节点及其子节点"""
            if node_idx in build_visited:
                # 如果已访问，返回基本信息（避免循环）
                return {
                    "id": node_idx,
                    "name": self.g.node_names[node_idx],
                    "type": self.g.node_types[node_idx],
                    "depth": node_info[node_idx]["depth"],
                    "is_leaf": node_info[node_idx]["is_leaf"],
                    "children": [],
                    "note": "cycle_detected",
                }
            build_visited.add(node_idx)
            
            info = node_info[node_idx].copy()
            
            # 优化2: 按关系和类型分组子节点，合并相同类型和关系的节点名称
            # 重点关注关系的多样性：每个不同的关系都会保留
            # 结构: {(relation, type): [child_node_idx, ...]}
            children_by_rel_type = {}
            for child_idx, rel_label in parent_children.get(node_idx, []):
                child_type = node_info[child_idx]["type"]
                key = (rel_label, child_type)
                if key not in children_by_rel_type:
                    children_by_rel_type[key] = []
                children_by_rel_type[key].append(child_idx)
            
            # 构建children列表，重点关注关系的多样性
            # 每个不同的关系都会创建一个子节点（即使类型相同）
            children = []
            for (rel_label, child_type), child_indices in children_by_rel_type.items():
                if len(child_indices) == 1:
                    # 只有一个节点，直接添加（name保持字符串）
                    child_node = build_node(child_indices[0])
                    child_node["relation"] = rel_label
                    children.append(child_node)
                else:
                    # 多个相同类型和关系的节点，合并名称到数组
                    child_names = [node_info[idx]["name"] for idx in child_indices]
                    # 使用第一个节点作为代表构建子树结构
                    representative = build_node(child_indices[0])
                    representative["relation"] = rel_label
                    representative["name"] = child_names  # 改为数组，减少重复
                    representative["count"] = len(child_indices)  # 记录合并的节点数量
                    # 注意：合并后的节点共享相同的children结构（来自第一个节点）
                    children.append(representative)
            
            info["children"] = children
            return info
        
        root_node = build_node(root_idx)
        
        # 更新根节点的is_leaf标记：如果根节点是目标类型，标记为叶子节点
        root_type = self.g.node_types[root_idx]
        root_type_lower = root_type.lower() if root_type else ""
        root_is_target = (root_type in target_types_set) or (root_type_lower in target_types_lower)
        if root_is_target:
            root_node["is_leaf"] = True
            node_info[root_idx]["is_leaf"] = True  # 同步更新node_info中的标记
        
        # 统计信息（优化后）
        total_nodes = len(node_info)
        total_edges = sum(len(children) for children in parent_children.values())
        leaf_count = sum(1 for node in node_info.values() if node.get("is_leaf", False))
        
        # 统计关系的多样性
        unique_relations = set()
        for children_list in parent_children.values():
            for _, rel_label in children_list:
                unique_relations.add(rel_label)
        relation_diversity = len(unique_relations)
        
        return {
            "root": root_node,
            "metadata": {
                "source": source,
                "source_id": root_idx,
                "target_types": target_types,
                "max_depth": max_depth,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "leaf_count": leaf_count,
                "expansions": expansions,
                "relation_diversity": relation_diversity,  # 关系多样性统计
                "unique_relations": sorted(list(unique_relations)),  # 所有唯一关系列表
                "debug": debug_info,
            }
        }
    
    def save_tree_to_json(self, tree_data: Dict[str, Any], output_path: str, indent: int = 2):
        """
        将树形结构保存为JSON文件。
        
        - tree_data: build_tree_from_source 返回的树形数据
        - output_path: 输出文件路径
        - indent: JSON缩进（默认2）
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=indent)
        print(f"树形结构已保存到: {output_path}")


def demo():
    pq = PathQuery()
    
    # 示例1: 构建树形结构
    print("=" * 60)
    print("示例1: 从 Cimetidine 构建树形结构（目标类型：Drug）")
    print("=" * 60)
    
    def progress_handler(info):
        print(f"  进度: 扩展={info['expansions']}, 队列={info['queue']}, 已找到节点={info['found']}, 深度={info['depth']}")
    
    tree_data = pq.build_tree_from_source(
        source="cimetidine",
        target_types=["Drug"],
        max_depth=4,  # 树的最大深度
        direction="both",
        use_display=True,
        branch_limit=150,
        max_expansions=500_000,
        progress_cb=progress_handler,
        allow_cycles=False,
    )
    
    metadata = tree_data["metadata"]
    print(f"\n构建完成:")
    print(f"  总节点数: {metadata['total_nodes']}")
    print(f"  总边数: {metadata['total_edges']}")
    print(f"  叶子节点数: {metadata['leaf_count']}")
    print(f"  扩展次数: {metadata['expansions']}")
    
    # 保存为JSON
    output_path = "cimetidine_tree.json"
    pq.save_tree_to_json(tree_data, output_path)
    
    # 示例2: 查看树的前几层结构
    print("\n" + "=" * 60)
    print("示例2: 树结构预览（前2层）")
    print("=" * 60)
    root = tree_data["root"]
    print(f"根节点: {root['name']} ({root['type']})")
    print(f"  深度: {root['depth']}, 叶子: {root['is_leaf']}, 子节点数: {len(root['children'])}")
    for i, child in enumerate(root['children'][:3]):  # 只显示前3个子节点
        print(f"  [{i+1}] {child['name']} ({child['type']}) -[{child.get('relation', 'N/A')}]->")
        print(f"      深度: {child['depth']}, 叶子: {child['is_leaf']}, 子节点数: {len(child['children'])}")
    
    # 示例3: 查询路径（旧功能，保留兼容）
    print("\n" + "=" * 60)
    print("示例3: 路径查询（兼容旧功能）")
    print("=" * 60)
    paths = pq.explore_from_source(
        "cimetidine",
        target_types=["Drug"],
        min_hops=2,
        max_hops=4,
        k=5,
        diversify=True,
    )
    print(f"找到 {len(paths)} 条路径:\n")
    for i, p in enumerate(paths[:3], 1):  # 只显示前3条
        print(f"{i}. {p['text']}")


if __name__ == "__main__":
    demo()

