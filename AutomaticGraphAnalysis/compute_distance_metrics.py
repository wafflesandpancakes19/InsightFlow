import pandas as pd
import networkx as nx
import numpy as np
import ot
from scipy.stats import entropy, wasserstein_distance
import os
import re
import pickle 

def extract_base_id(filename):
    """
    Extract the leading integer ID from filenames of the format '123_out...'.
    Returns None if no ID is found.
    """
    match = re.match(r"(\d+)_out", filename)
    return match.group(1) if match else None

def load_graph_from_excel(filepath):
    """
    Load a graph from an Excel file with columns: 'Node Code', 'Nodes', 'Edges 1', 'Edges 2'.
    Returns a NetworkX DiGraph.
    """
    df = pd.read_excel(filepath)

    # Map node codes → node labels
    node_map = {
        str(k).strip(): str(v).strip()
        for k, v in zip(df["Node Code"], df["Nodes"])
        if pd.notna(k) and pd.notna(v)
    }

    G = nx.DiGraph()
    G.add_nodes_from(node_map.values())

    for _, row in df.iterrows():
        src = row.get("Edges 1")
        tgt = row.get("Edges 2")

        if pd.notna(src) and pd.notna(tgt):
            src_label = node_map.get(str(src).strip())
            tgt_label = node_map.get(str(tgt).strip())

            if src_label and tgt_label:
                G.add_edge(src_label, tgt_label)

    return G

def load_graph_from_gpickle(filepath):
    """
    Load a graph from a gpickle file (NetworkX 3.x compatible).
    """
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    return G

def load_graph(filepath):
    """
    Load a graph from Excel (.xlsx, .xls) or gpickle (.gpickle, .pkl) file.
    Returns a NetworkX graph.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return load_graph_from_excel(filepath)
    elif ext in [".gpickle", ".pkl"]:
        return load_graph_from_gpickle(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def degree_distribution(graph, max_degree=None):
    """
    Compute the normalized degree distribution of a graph.
    """
    degrees = [d for _, d in graph.degree()]

    if max_degree is None:
        max_degree = max(degrees) if degrees else 0

    hist = np.zeros(max_degree + 1)
    for d in degrees:
        hist[d] += 1

    prob = hist / (hist.sum() if hist.sum() > 0 else 1)
    return prob

def kl_divergence(p, q, eps=1e-10):
    """
    Compute KL divergence between two probability distributions.
    """
    p = p + eps
    q = q + eps
    return entropy(p, q)

def earth_movers_distance(p, q):
    """
    Compute Earth Mover's Distance using OT library.
    """
    n = len(p)
    positions = np.arange(n).reshape(-1, 1)
    M = ot.dist(positions, positions, metric='euclidean')
    return float(ot.emd2(p, q, M))

def wasserstein_dist(p, q):
    """
    Compute 1D Wasserstein distance.
    """
    bins = np.arange(len(p))
    return wasserstein_distance(bins, bins, p, q)

def compare_graph_folders(folder_a, folder_b, output_excel="graph_distances.xlsx"):
    """
    Compare all graphs in folder_a with corresponding graphs in folder_b (matching by base ID).
    Computes KL divergence, EMD, and Wasserstein distance based on degree distributions.
    """
    # Map filenames by base ID
    files_a = {extract_base_id(f): os.path.join(folder_a, f) 
               for f in os.listdir(folder_a) if extract_base_id(f)}
    files_b = {extract_base_id(f): os.path.join(folder_b, f) 
               for f in os.listdir(folder_b) if extract_base_id(f)}

    common_ids = sorted(set(files_a.keys()) & set(files_b.keys()))
    if not common_ids:
        print("No matching files found by ID.")
        return

    results = []

    for file_id in common_ids:
        path_a = files_a[file_id]
        path_b = files_b[file_id]

        print(f"\nComparing:\n  {os.path.basename(path_a)}\n  {os.path.basename(path_b)}")

        G1 = load_graph(path_a)
        G2 = load_graph(path_b)

        # Degree distributions
        max_deg = max(
            max(dict(G1.degree()).values(), default=0),
            max(dict(G2.degree()).values(), default=0)
        )

        p = degree_distribution(G1, max_deg)
        q = degree_distribution(G2, max_deg)

        # Metrics
        KL = kl_divergence(p, q)
        EMD = earth_movers_distance(p, q)
        WD = wasserstein_dist(p, q)

        results.append({
            "file_A": os.path.basename(path_a),
            "file_B": os.path.basename(path_b),
            "KL_divergence": KL,
            "EMD": EMD,
            "Wasserstein": WD
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nSaved results to {output_excel}")

compare_graph_folders("avani", "minoti", output_excel=r"automatic_graph_analysis\human\graph_distances_avani_minoti.xlsx")
compare_graph_folders("bhavyaa", "saniya", output_excel=r"automatic_graph_analysis\human\graph_distances_bhavyaa_saniya.xlsx")
compare_graph_folders("avani", "gpickles_1", output_excel=r"automatic_graph_analysis\llm\graph_distances_avani_auto.xlsx")
compare_graph_folders("minoti", "gpickles_1", output_excel=r"automatic_graph_analysis\llm\graph_distances_minoti_auto.xlsx")
compare_graph_folders("bhavyaa", "gpickles_2", output_excel=r"automatic_graph_analysis\llm\graph_distances_bhavyaa_auto.xlsx")
compare_graph_folders("saniya", "gpickles_2", output_excel=r"automatic_graph_analysis\llm\graph_distances_saniya_auto.xlsx")