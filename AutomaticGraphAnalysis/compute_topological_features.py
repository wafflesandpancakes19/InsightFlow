import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx


# ------------------------------------------
# Load graph from Excel
# ------------------------------------------
def load_graph_from_excel(filepath):
    df = pd.read_excel(filepath)
    node_map = {
        str(k).strip(): str(v).strip()
        for k, v in zip(df["Node Code"], df["Nodes"])
        if pd.notna(k) and pd.notna(v)
    }

    edges = []
    for _, row in df.iterrows():
        src = row["Edges 1"]
        tgt = row["Edges 2"]
        if pd.notna(src) and pd.notna(tgt):
            src_text = node_map.get(str(src).strip())
            tgt_text = node_map.get(str(tgt).strip())
            if src_text and tgt_text:
                edges.append((src_text, tgt_text))

    return edges, list(set(node_map.values()))


# ------------------------------------------
# Load graph from gpickle
# ------------------------------------------
def load_graph_from_gpickle(filepath):
    with open(filepath, "rb") as f:
        G = pickle.load(f)
    return list(G.edges()), list(G.nodes())


# ------------------------------------------
# Auto-detect loader
# ------------------------------------------
def load_graph(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".gpickle", ".gpkl"]:
        return load_graph_from_gpickle(filepath)

    elif ext in [".xlsx", ".xls"]:
        return load_graph_from_excel(filepath)

    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    

# ------------------------------------------
# Your Topological Feature Computation
# ------------------------------------------
def compute_topological_features(G):
    features = {}

    # Handle empty graph
    if G.number_of_nodes() == 0:
        return {
            "diameter": np.nan,
            "avg_shortest_path": np.nan,
            "clustering": np.nan,
            "density": np.nan,
            "assortativity": np.nan,
            "transitivity": np.nan,
        }

    # Diameter & shortest path
    if nx.is_connected(G):
        GC = G
    else:
        GC = G.subgraph(max(nx.connected_components(G), key=len))

    try:
        features["diameter"] = nx.diameter(GC)
        features["avg_shortest_path"] = nx.average_shortest_path_length(GC)
    except:
        features["diameter"] = np.nan
        features["avg_shortest_path"] = np.nan

    # Clustering
    try:
        features["clustering"] = nx.average_clustering(G)
    except:
        features["clustering"] = np.nan

    # Density
    try:
        features["density"] = nx.density(G)
    except:
        features["density"] = np.nan

    # Assortativity
    try:
        features["assortativity"] = nx.degree_assortativity_coefficient(G)
    except:
        features["assortativity"] = np.nan

    # Transitivity
    try:
        features["transitivity"] = nx.transitivity(G)
    except:
        features["transitivity"] = np.nan

    return features


# ------------------------------------------
# Compute features for a folder
# ------------------------------------------
def compute_features_for_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if not os.path.isfile(full_path):
            continue

        try:
            edges, nodes = load_graph(full_path)

            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            features = compute_topological_features(G)
            features["filename"] = file
            features["error"] = ""

        except Exception as e:
            features = {
                "filename": file,
                "diameter": np.nan,
                "avg_shortest_path": np.nan,
                "clustering": np.nan,
                "density": np.nan,
                "assortativity": np.nan,
                "transitivity": np.nan,
                "error": str(e)
            }

        results.append(features)

    return pd.DataFrame(results)


def process_single_folder(folder, output_excel="topology_features.xlsx"):
    print(f"Processing folder: {folder}")
    
    df = compute_features_for_folder(folder)
    df.insert(0, "folder", os.path.basename(folder)) 

    # Summary rows
    numeric_cols = [
        "diameter", "avg_shortest_path", "clustering",
        "density", "assortativity", "transitivity"
    ]

    mean_row = {col: df[col].mean() for col in numeric_cols}
    std_row = {col: df[col].std() for col in numeric_cols}

    mean_row.update({"folder": "", "filename": "MEAN", "error": ""})
    std_row.update({"folder": "", "filename": "STD", "error": ""})

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    df.to_excel(output_excel, index=False)
    print(f"Saved to {output_excel}")


process_single_folder(folder="avani", output_excel=r"automatic_graph_analysis\human\avani_topo_features.xlsx")
process_single_folder(folder="bhavyaa", output_excel=r"automatic_graph_analysis\human\bhavyaa_topo_features.xlsx")
process_single_folder(folder="minoti", output_excel=r"automatic_graph_analysis\human\minoti_topo_features.xlsx")
process_single_folder(folder="saniya", output_excel=r"automatic_graph_analysis\human\saniya_topo_features.xlsx")
process_single_folder(folder="gpickles_1", output_excel=r"automatic_graph_analysis\llm\group_1_llm_topo_features.xlsx")
process_single_folder(folder="gpickles_2", output_excel=r"automatic_graph_analysis\llm\group_2_llm_topo_features.xlsx")