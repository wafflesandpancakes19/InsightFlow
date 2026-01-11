import os
import re
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
import pickle
from sklearn.cluster import SpectralClustering
from networkx.algorithms.community import girvan_newman, label_propagation_communities
from infomap import Infomap

# --------------------- Graph Loading ---------------------
def extract_base_id(filename):
    """Extract leading integer ID from filename."""
    match = re.match(r"(\d+)_out", filename)
    return match.group(1) if match else None

def load_graph_from_excel(filepath):
    """Load NetworkX DiGraph from Excel file."""
    df = pd.read_excel(filepath)
    node_map = {
        str(k).strip(): str(v).strip()
        for k, v in zip(df["Node Code"], df["Nodes"])
        if pd.notna(k) and pd.notna(v)
    }
    G = nx.DiGraph()
    G.add_nodes_from(node_map.values())
    for _, row in df.iterrows():
        src, tgt = row.get("Edges 1"), row.get("Edges 2")
        if pd.notna(src) and pd.notna(tgt):
            src_label = node_map.get(str(src).strip())
            tgt_label = node_map.get(str(tgt).strip())
            if src_label and tgt_label:
                G.add_edge(src_label, tgt_label)
    return G

def load_graph_from_gpickle(filepath):
    """Load NetworkX graph from gpickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_graph(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return load_graph_from_excel(filepath)
    elif ext in [".gpickle", ".pkl"]:
        return load_graph_from_gpickle(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

# --------------------- Community Detection ---------------------
def leiden_communities(G):
    if len(G.nodes()) == 0:
        return []
    ig_G = ig.Graph.from_networkx(G)
    partition = la.find_partition(ig_G, la.ModularityVertexPartition)
    return [set(c) for c in partition]

def louvain_communities(G):
    import community.community_louvain as louvain
    if len(G.nodes()) == 0:
        return []
    if len(G.edges()) == 0:
        return [set([n]) for n in G.nodes()]
    # Ensure nodes are strings
    H = nx.relabel_nodes(G, lambda n: str(n))
    partition = louvain.best_partition(H)
    communities = {}
    for node, c in partition.items():
        communities.setdefault(c, set()).add(node)
    return list(communities.values())

def girvan_newman_communities(G, k=2):
    G_undirected = G.to_undirected()
    if len(G_undirected.nodes()) == 0:
        return []
    communities_gen = girvan_newman(G_undirected)
    try:
        for _ in range(k-1):
            next(communities_gen)
        communities = tuple(sorted(c) for c in next(communities_gen))
    except StopIteration:
        communities = tuple(sorted(c) for c in next(communities_gen))
    return [set(c) for c in communities]

def label_propagation_communities_method(G):
    G_undirected = G.to_undirected()
    if len(G_undirected.nodes()) == 0:
        return []
    communities = list(label_propagation_communities(G_undirected))
    return [set(c) for c in communities]

def spectral_communities(G, n_clusters=2):
    G_undirected = G.to_undirected()
    if len(G_undirected.nodes()) == 0:
        return []
    A = nx.to_numpy_array(G_undirected)
    n_clusters = min(n_clusters, len(G_undirected.nodes()))
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(A)
    communities = {}
    nodes_list = list(G_undirected.nodes())
    for node, lab in enumerate(labels):
        communities.setdefault(lab, set()).add(nodes_list[node])
    return list(communities.values())

def infomap_communities(G):
    G_undirected = G.to_undirected()
    if len(G_undirected.nodes()) == 0:
        return []
    # Map nodes to integers
    node_to_id = {node: i for i, node in enumerate(G_undirected.nodes())}
    id_to_node = {i: node for node, i in node_to_id.items()}

    im = Infomap()
    for u, v in G_undirected.edges():
        im.add_link(node_to_id[u], node_to_id[v])
    im.run()

    communities = {}
    for node in im.nodes:
        node_label = id_to_node[node.node_id]
        communities.setdefault(node.module_id, set()).add(node_label)
    return list(communities.values())

# --------------------- Folder Processing ---------------------
def process_folder(folder_path, output_excel="graph_communities.xlsx"):
    results = []

    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        graph_id = extract_base_id(filename) or filename
        try:
            G = load_graph(filepath)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        print(f"Processing graph: {filename} ({len(G.nodes())} nodes)")

        row = {"graph_id": graph_id}
        try:
            row["leiden"] = str(leiden_communities(G))
        except Exception as e:
            row["leiden"] = f"Error: {e}"
        try:
            row["louvain"] = str(louvain_communities(G))
        except Exception as e:
            row["louvain"] = f"Error: {e}"
        try:
            row["girvan_newman"] = str(girvan_newman_communities(G))
        except Exception as e:
            row["girvan_newman"] = f"Error: {e}"
        try:
            row["label_propagation"] = str(label_propagation_communities_method(G))
        except Exception as e:
            row["label_propagation"] = f"Error: {e}"
        try:
            row["spectral"] = str(spectral_communities(G))
        except Exception as e:
            row["spectral"] = f"Error: {e}"
        try:
            row["infomap"] = str(infomap_communities(G))
        except Exception as e:
            row["infomap"] = f"Error: {e}"

        results.append(row)

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nAll communities saved to {output_excel}")

# --------------------- Example Usage ---------------------
process_folder("avani", output_excel=r"automatic_graph_analysis\human\avani_communities.xlsx")
process_folder("bhavyaa", output_excel=r"automatic_graph_analysis\human\bhavyaa_communities.xlsx")
process_folder("minoti", output_excel=r"automatic_graph_analysis\human\minoti_communities.xlsx")
process_folder("saniya", output_excel=r"automatic_graph_analysis\human\saniya_communities.xlsx")
process_folder("gpickles_1", output_excel=r"automatic_graph_analysis\llm\group_1_auto_communities.xlsx")
process_folder("gpickles_2", output_excel=r"automatic_graph_analysis\llm\group_2_auto_communities.xlsx")
