import os
import re
import pandas as pd
import networkx as nx
import pickle
import numpy as np

def extract_base_id(filename):
    """Extract the leading integer ID from filename, e.g., '123_out.xlsx' → '123'."""
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
    """Load a NetworkX graph from a gpickle file (NetworkX 3.x compatible)."""
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    return G

def load_graph(filepath):
    """Load a graph from Excel or gpickle."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return load_graph_from_excel(filepath)
    elif ext in [".gpickle", ".pkl"]:
        return load_graph_from_gpickle(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def compute_centralities(G):
    """Compute degree, betweenness, and closeness centralities."""
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, normalized=True)
    closeness_c = nx.closeness_centrality(G)

    # Combine into a single dictionary
    centralities = {}
    for node in G.nodes():
        centralities[node] = {
            "degree": degree_c.get(node, 0),
            "betweenness": betweenness_c.get(node, 0),
            "closeness": closeness_c.get(node, 0)
        }
    return centralities

def process_folder(folder_path, output_excel="centralities.xlsx"):
    """
    Process all Excel or gpickle files in a folder, compute centralities, 
    and save node-level and graph-level summary statistics.
    """
    node_results = []
    summary_results = []

    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        base_id = extract_base_id(filename) or filename

        try:
            G = load_graph(filepath)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        centralities = compute_centralities(G)

        # Node-level results
        for node, metrics in centralities.items():
            node_results.append({
                "graph_id": base_id,
                "node": node,
                "degree_centrality": metrics["degree"],
                "betweenness_centrality": metrics["betweenness"],
                "closeness_centrality": metrics["closeness"]
            })

        # Graph-level summary statistics
        metrics_df = pd.DataFrame(centralities).T
        summary_results.append({
            "graph_id": base_id,
            "degree_mean": metrics_df["degree"].mean(),
            "degree_std": metrics_df["degree"].std(),
            "degree_min": metrics_df["degree"].min(),
            "degree_max": metrics_df["degree"].max(),
            "betweenness_mean": metrics_df["betweenness"].mean(),
            "betweenness_std": metrics_df["betweenness"].std(),
            "betweenness_min": metrics_df["betweenness"].min(),
            "betweenness_max": metrics_df["betweenness"].max(),
            "closeness_mean": metrics_df["closeness"].mean(),
            "closeness_std": metrics_df["closeness"].std(),
            "closeness_min": metrics_df["closeness"].min(),
            "closeness_max": metrics_df["closeness"].max()
        })

        print(f"Processed graph: {filename} ({len(G.nodes())} nodes)")

    # Save node-level and summary statistics to separate sheets
    with pd.ExcelWriter(output_excel) as writer:
        pd.DataFrame(node_results).to_excel(writer, sheet_name="node_centralities", index=False)
        pd.DataFrame(summary_results).to_excel(writer, sheet_name="graph_summary", index=False)

    print(f"\nCentralities saved to {output_excel}")

# Example usage:
#process_folder("avani", output_excel=r"automatic_graph_analysis\human\avani_centralities.xlsx")
process_folder("bhavyaa", output_excel=r"automatic_graph_analysis\human\bhavyaa_centralities_new.xlsx")
#process_folder("minoti", output_excel=r"automatic_graph_analysis\human\minoti_centralities.xlsx")
#process_folder("saniya", output_excel=r"automatic_graph_analysis\human\saniya_centralities.xlsx")
#process_folder("gpickles_1", output_excel=r"automatic_graph_analysis\llm\group_1_auto_centralities.xlsx")
#process_folder("gpickles_2", output_excel=r"automatic_graph_analysis\llm\group_2_auto_centralities.xlsx")
