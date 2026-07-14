from data_process.build_prime_kg import load_query_graph
from data_process.paths import PRIMEKG_GRAPH_PKL

g = load_query_graph(str(PRIMEKG_GRAPH_PKL))
print('nodes', len(g.node_names))
print('edges', len(g.edge_index_list))
# pick a sample neighbor to test path
nb = g.neighbors('OR2J3', direction='out')
print('nb_count', len(nb))
if nb:
    _, dst_idx, rel = nb[0]
    dst_name = g.node_names[dst_idx]
    steps = g.shortest_path_with_edges('OR2J3', dst_name, max_hops=2, direction='both')
    print('path_len', len(steps))
    print(g.format_path(steps))