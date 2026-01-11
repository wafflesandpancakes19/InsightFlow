import os
import re
import pandas as pd
import networkx as nx
import pickle
import numpy as np

def extract_base_id(filename):
    """Extract leading integer ID from filename (e.g., '123_out.xlsx' → '123')."""
    match = re.match(r"(\d+)_out", filename)
    return match.group(1) if match else None

def load_graph_from_excel(filepath):
    """Load a NetworkX DiGraph from an Excel file."""
    df = pd.read_excel(filepath)
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
    """Load a NetworkX graph from a gpickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_graph(filepath):
    """Load a graph from Excel or gpickle."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return load_graph_from_excel(filepath)
    elif ext in [".gpickle", ".pkl"]:
        return load_graph_from_gpickle(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def compute_graph_metrics(G):
    """Compute graph-level clustering and triangle metrics (works for directed graphs)."""
    
    # Convert to undirected graph for clustering/triangle functions
    G_undirected = G.to_undirected()
    
    local_clustering = nx.clustering(G_undirected)
    triangles = nx.triangles(G_undirected)
    square_clustering = nx.square_clustering(G_undirected)

    node_df = pd.DataFrame({
        "local_clustering": list(local_clustering.values()),
        "triangles": list(triangles.values()),
        "square_clustering": list(square_clustering.values())
    })

    metrics = {
        "avg_local_clustering": node_df["local_clustering"].mean(),
        "std_local_clustering": node_df["local_clustering"].std(),
        "min_local_clustering": node_df["local_clustering"].min(),
        "max_local_clustering": node_df["local_clustering"].max(),
        "avg_triangles": node_df["triangles"].mean(),
        "std_triangles": node_df["triangles"].std(),
        "min_triangles": node_df["triangles"].min(),
        "max_triangles": node_df["triangles"].max(),
        "avg_square_clustering": node_df["square_clustering"].mean(),
        "std_square_clustering": node_df["square_clustering"].std(),
        "min_square_clustering": node_df["square_clustering"].min(),
        "max_square_clustering": node_df["square_clustering"].max(),
        "avg_clustering": nx.average_clustering(G_undirected),
        "transitivity": nx.transitivity(G_undirected)
    }

    return metrics

def process_folder(folder_path, output_excel="graph_clustering_metrics.xlsx"):
    """
    Process all Excel or gpickle files in a folder, compute graph-level metrics,
    and save results to Excel (one row per graph).
    """
    results = []

    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        graph_id = extract_base_id(filename) or filename

        try:
            G = load_graph(filepath)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        metrics = compute_graph_metrics(G)
        metrics["graph_id"] = graph_id
        results.append(metrics)

        print(f"Processed graph: {filename} ({len(G.nodes())} nodes)")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nGraph-level metrics saved to {output_excel}")


#process_folder("avani", output_excel=r"automatic_graph_analysis\human\avani_clustering.xlsx")
process_folder("bhavyaa", output_excel=r"automatic_graph_analysis\human\bhavyaa_clustering_new.xlsx")
#process_folder("minoti", output_excel=r"automatic_graph_analysis\human\minoti_clustering.xlsx")
#process_folder("saniya", output_excel=r"automatic_graph_analysis\human\saniya_clustering.xlsx")
#process_folder("gpickles_1", output_excel=r"automatic_graph_analysis\llm\group_1_auto_clustering.xlsx")
#process_folder("gpickles_2", output_excel=r"automatic_graph_analysis\llm\group_2_auto_clustering.xlsx")