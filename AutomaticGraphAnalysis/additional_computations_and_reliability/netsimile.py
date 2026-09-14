import numpy as np
import networkx as nx

def _safe_stat(func, array):
    try:
        return func(array)
    except Exception:
        return 0.0

def _aggregate_node_features(feature_matrix, aggregators):
    return np.concatenate([
        np.array([_safe_stat(func, feature_matrix[:, i]) for func in aggregators])
        for i in range(feature_matrix.shape[1])
    ])

def compute_node_features(G):
    degrees = dict(G.degree())
    clustering = nx.clustering(G) if not G.is_directed() else nx.clustering(G.to_undirected())
    triangles = nx.triangles(G) if not G.is_directed() else nx.triangles(G.to_undirected())
    avg_neighbor_deg = nx.average_neighbor_degree(G)

    node_features = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node)) if G.is_directed() else list(G.adj[node])
        ego = nx.ego_graph(G, node, radius=1, center=True)
        ego_edges = ego.number_of_edges()
        out_links = len(set(G.neighbors(node)) - set(ego.nodes())) if G.is_directed() else 0

        features = [
            degrees.get(node, 0),
            clustering.get(node, 0.0),
            np.mean([degrees.get(n, 0) for n in neighbors]) if neighbors else 0.0,
            ego_edges,
            out_links,
            triangles.get(node, 0),
            avg_neighbor_deg.get(node, 0.0),
        ]
        node_features.append(features)

    return np.array(node_features)

def netsimile_features(
    G,
    aggregators=None,
    normalize=True
):
    if aggregators is None:
        aggregators = [np.mean, np.median, np.std, np.min, np.max, 
                       lambda x: np.percentile(x, 25), 
                       lambda x: np.percentile(x, 75)]

    if len(G.nodes) == 0:
        return np.zeros(7 * len(aggregators))  # 7 features × #aggregators

    node_features = compute_node_features(G)

    if normalize:
        # Normalize each feature by dividing by max (avoiding division by zero)
        max_vals = np.max(node_features, axis=0)
        max_vals[max_vals == 0] = 1
        node_features = node_features / max_vals

    return _aggregate_node_features(node_features, aggregators)
