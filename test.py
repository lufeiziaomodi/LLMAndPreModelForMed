from prime_kg_utils.path_query import PathQuery
from data_process.paths import DATA_REPORTS, ensure_dir

# 使用默认 KG 图路径 (data/kg/primekg_graph.pkl)
pq = PathQuery()


tree_data = pq.build_tree_from_source(
    source="cimetidine",
    target_types=["drug"],  # 药物作为叶子节点
    max_depth=2,            # 树的最大深度
    direction="both",
    use_display=True,
    branch_limit=150,       # 每个节点最多扩展150个邻居
    max_expansions=500_000, # 最多扩展50万次
    progress_cb=lambda info: print(f"进度: {info}"),
    allow_cycles=False,     # 不允许循环
)

# 保存为JSON（demo 输出落 data/reports/legacy_demos/）
demo_dir = ensure_dir(DATA_REPORTS / "legacy_demos")
pq.save_tree_to_json(tree_data, str(demo_dir / "cimetidine_tree.json"))

# 查看统计信息
metadata = tree_data["metadata"]
print(f"总节点数: {metadata['total_nodes']}")
print(f"叶子节点数: {metadata['leaf_count']}")


# # 目标已知，带进度回调
# paths = pq.find_paths_by_name(
#     "cimetidine", "hypertension",
#     min_hops=3, max_hops=5, k=5,
#     progress_cb=lambda info: print("progress", info)
# )
# print(pq)
# print("hello")
# print(paths)
# 从源出发，目标为疾病类型，限制长度

# paths = pq.find_paths_by_name(
#     "cimetidine", "iodamide",
#     min_hops=3, max_hops=6,
#     k=50,
#     branch_limit=150,         # 可调小防爆
#     max_expansions=500_000,   # 控制时间/内存
#     progress_cb=lambda info: print(info)
# )

# for p in paths:
#     print(p["text"])

# paths1 = pq.explore_from_source(
#     "cimetidine",
#     target_types=["drug"],
#     min_hops=2,
#     max_hops=6,
#     k=10,
#     diversify=True,        # 启用多样性过滤
#     max_per_group=2,       # 每组最多保留2条不同终点的路径
#     search_more=3,         # 搜索30条路径后过滤出10条多样化路径
# )
#
#
# paths2 = pq.explore_from_source(
#     "iodamide",
#     target_types=["drug"],
#     min_hops=2,
#     max_hops=6,
#     k=10,
#     diversify=True,        # 启用多样性过滤
#     max_per_group=2,       # 每组最多保留2条不同终点的路径
#     search_more=3,         # 搜索30条路径后过滤出10条多样化路径
# )
#
#
#
# print("---------------------------")
#
# for p in paths1:
#     print(p["text"])
#
# print("---------------------------")
# for p in paths2:
#     print(p["text"])
