import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from netsimile import netsimile_features  # You should have your own version
import pickle

# Config
FOLDER_A = "saniya2"
FOLDER_B = "bhavyaa2"
AUTO_FOLDER = "gpickles-2"  # new
OUTPUT_CSV = "graph_agreement_results_with_auto_g2.csv"
SUMMARY_CSV = "graph_agreement_summary_with_auto_g2.csv"
FIGURE_DIR = "visualizations_fin_g1"
SIM_THRESHOLD = 0.6
DIRECTED_GRAPH = False

os.makedirs(FIGURE_DIR, exist_ok=True)
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_graph_from_excel(filepath):
    df = pd.read_excel(filepath)
    node_map = {str(k).strip(): str(v).strip()
                for k, v in zip(df["Node Code"], df["Nodes"])
                if pd.notna(k) and pd.notna(v)}
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


def compute_node_matches(nodes_a, nodes_b, threshold=0.6):
    emb_a = model.encode(nodes_a, convert_to_tensor=True)
    emb_b = model.encode(nodes_b, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb_a, emb_b)
    matched = {}
    used_b = set()
    for i, sim_row in enumerate(sim_matrix):
        j = torch.argmax(sim_row).item()
        if sim_row[j] >= threshold and j not in used_b:
            matched[nodes_a[i]] = nodes_b[j]
            used_b.add(j)
        else:
            matched[nodes_a[i]] = nodes_a[i]
    for b in nodes_b:
        if b not in matched.values():
            matched[b] = b
    return matched


def normalize_edges(edges, mapping, directed=True):
    norm_edges = set()
    for src, tgt in edges:
        src_mapped = mapping.get(src, src)
        tgt_mapped = mapping.get(tgt, tgt)
        if not directed:
            norm_edges.add(tuple(sorted((src_mapped, tgt_mapped))))
        else:
            norm_edges.add((src_mapped, tgt_mapped))
    return norm_edges


def build_graph_from_edges(edges):
    G = nx.DiGraph() if DIRECTED_GRAPH else nx.Graph()
    G.add_edges_from(edges)
    return G


def compute_netsimile_similarity(edges_a, edges_b):
    G_a = build_graph_from_edges(edges_a)
    G_b = build_graph_from_edges(edges_b)
    try:
        f_a = netsimile_features(G_a)
        f_b = netsimile_features(G_b)
        dist = euclidean(f_a, f_b)
        return 1 / (1 + dist)
    except Exception as e:
        print(f"NetSimile error: {e}")
        return 0.0


def plot_edge_diff_graph(edges_a, edges_b, file_name):
    G = nx.Graph()
    only_a = edges_a - edges_b
    only_b = edges_b - edges_a
    both = edges_a & edges_b
    edge_colors = []
    for e in only_a:
        G.add_edge(*e)
        edge_colors.append('red')
    for e in only_b:
        G.add_edge(*e)
        edge_colors.append('blue')
    for e in both:
        G.add_edge(*e)
        edge_colors.append('green')
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors,
            node_color='lightgray', font_size=8, width=2)
    plt.title("Edge Differences (Red: A only, Blue: B only, Green: Both)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"{file_name}_edge_diff.png"))
    plt.close()


def plot_radar_chart(metrics_dict, file_name):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    plt.title("Graph Similarity Metrics Radar")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"{file_name}_radar.png"))
    plt.close()

# --- New Semantic Similarity Utilities ---

def compute_soft_jaccard(edges_a, edges_b, model, threshold=0.6):
    """Soft Jaccard similarity between edges using SBERT embeddings."""
    if not edges_a or not edges_b:
        return 0.0
    edge_texts_a = [" ".join(e) for e in edges_a]
    edge_texts_b = [" ".join(e) for e in edges_b]
    emb_a = model.encode(edge_texts_a, convert_to_tensor=True)
    emb_b = model.encode(edge_texts_b, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(emb_a, emb_b).cpu().numpy()

    matches = []
    for i in range(len(edge_texts_a)):
        best_sim = np.max(sims[i])
        if best_sim >= threshold:
            matches.append(best_sim)
    if not matches:
        return 0.0
    return float(np.mean(matches))


def compute_node_set_similarity(nodes_a, nodes_b, model):
    """Cosine similarity between centroids of node embeddings."""
    if not nodes_a or not nodes_b:
        return 0.0
    emb_a = model.encode(nodes_a, convert_to_tensor=True)
    emb_b = model.encode(nodes_b, convert_to_tensor=True)
    mean_a = torch.mean(emb_a, dim=0)
    mean_b = torch.mean(emb_b, dim=0)
    return util.pytorch_cos_sim(mean_a, mean_b).item()


def compute_node_centrality_similarity(edges_a, edges_b, nodes_a, nodes_b, model):
    """Similarity of central nodes weighted by degree centrality."""
    G_a = build_graph_from_edges(edges_a)
    G_b = build_graph_from_edges(edges_b)
    if not G_a.nodes() or not G_b.nodes():
        return 0.0

    centrality_a = nx.degree_centrality(G_a)
    centrality_b = nx.degree_centrality(G_b)

    emb_a = model.encode(list(G_a.nodes()), convert_to_tensor=True)
    emb_b = model.encode(list(G_b.nodes()), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(emb_a, emb_b).cpu().numpy()
    weighted_scores = []

    for i, node_a in enumerate(G_a.nodes()):
        j = np.argmax(sims[i])
        score = sims[i, j]
        weight = centrality_a[node_a] * centrality_b[list(G_b.nodes())[j]]
        weighted_scores.append(score * weight)

    if not weighted_scores:
        return 0.0
    return float(np.sum(weighted_scores) / (np.sum(list(centrality_a.values())) + 1e-6))


# --- Modified compare_and_record ---
def compare_and_record(edges1, nodes1, edges2, nodes2, file, label, results):
    mapping_1_to_2 = compute_node_matches(nodes1, nodes2, SIM_THRESHOLD)
    mapping_2_to_1 = compute_node_matches(nodes2, nodes1, SIM_THRESHOLD)
    merged_mapping = {**mapping_1_to_2, **{v: k for k, v in mapping_2_to_1.items()}}
    norm_1 = normalize_edges(edges1, merged_mapping, directed=DIRECTED_GRAPH)
    norm_2 = normalize_edges(edges2, merged_mapping, directed=DIRECTED_GRAPH)
    all_edges = list(norm_1.union(norm_2))
    vec1 = [1 if e in norm_1 else 0 for e in all_edges]
    vec2 = [1 if e in norm_2 else 0 for e in all_edges]

    if np.sum(np.logical_or(vec1, vec2)) == 0:
        precision = recall = f1 = jaccard = 0.0
    else:
        precision = (precision_score(vec1, vec2, zero_division=0) + precision_score(vec2, vec1, zero_division=0)) / 2
        recall = (recall_score(vec1, vec2, zero_division=0) + recall_score(vec2, vec1, zero_division=0)) / 2
        f1 = (f1_score(vec1, vec2, zero_division=0) + f1_score(vec2, vec1, zero_division=0)) / 2
        jaccard = np.sum(np.logical_and(vec1, vec2)) / np.sum(np.logical_or(vec1, vec2))
    netsim = compute_netsimile_similarity(norm_1, norm_2)

    # --- new metrics ---
    soft_jaccard = compute_soft_jaccard(norm_1, norm_2, model, SIM_THRESHOLD)
    node_set_sim = compute_node_set_similarity(nodes1, nodes2, model)
    node_centrality_sim = compute_node_centrality_similarity(norm_1, norm_2, nodes1, nodes2, model)

    results.append({
        "file": file,
        "comparison": label,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "jaccard_similarity": jaccard,
        "netsimile_similarity": netsim,
        "soft_jaccard_similarity": soft_jaccard,
        "node_set_similarity": node_set_sim,
        "node_centrality_similarity": node_centrality_sim,
        "edges_graph_1": len(norm_1),
        "edges_graph_2": len(norm_2),
        "matched_edges": sum(np.logical_and(vec1, vec2))
    })

    plot_edge_diff_graph(norm_1, norm_2, f"{os.path.splitext(file)[0]}_{label}")



def compare_and_record_old(edges1, nodes1, edges2, nodes2, file, label, results):
    mapping_1_to_2 = compute_node_matches(nodes1, nodes2, SIM_THRESHOLD)
    mapping_2_to_1 = compute_node_matches(nodes2, nodes1, SIM_THRESHOLD)
    merged_mapping = {**mapping_1_to_2, **{v: k for k, v in mapping_2_to_1.items()}}
    norm_1 = normalize_edges(edges1, merged_mapping, directed=DIRECTED_GRAPH)
    norm_2 = normalize_edges(edges2, merged_mapping, directed=DIRECTED_GRAPH)
    all_edges = list(norm_1.union(norm_2))
    vec1 = [1 if e in norm_1 else 0 for e in all_edges]
    vec2 = [1 if e in norm_2 else 0 for e in all_edges]

    if np.sum(np.logical_or(vec1, vec2)) == 0:
        precision = recall = f1 = jaccard = 0.0
    else:
        precision = (precision_score(vec1, vec2, zero_division=0) + precision_score(vec2, vec1, zero_division=0)) / 2
        recall = (recall_score(vec1, vec2, zero_division=0) + recall_score(vec2, vec1, zero_division=0)) / 2
        f1 = (f1_score(vec1, vec2, zero_division=0) + f1_score(vec2, vec1, zero_division=0)) / 2
        jaccard = np.sum(np.logical_and(vec1, vec2)) / np.sum(np.logical_or(vec1, vec2))
    netsim = compute_netsimile_similarity(norm_1, norm_2)

    results.append({
        "file": file,
        "comparison": label,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "jaccard_similarity": jaccard,
        "netsimile_similarity": netsim,
        "edges_graph_1": len(norm_1),
        "edges_graph_2": len(norm_2),
        "matched_edges": sum(np.logical_and(vec1, vec2))
    })

    plot_edge_diff_graph(norm_1, norm_2, f"{os.path.splitext(file)[0]}_{label}")
    
    """
    plot_radar_chart({
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Jaccard": jaccard,
        "NetSimile": netsim
    }, f"{os.path.splitext(file)[0]}_{label}")
    """


# === Main Comparison Loop ===
results = []
for file in sorted(os.listdir(FOLDER_A)):
    if not file.endswith(".xlsx"):
        continue

    base = os.path.splitext(file)[0]
    path_a = os.path.join(FOLDER_A, file)
    path_b = os.path.join(FOLDER_B, file)
    path_auto = os.path.join(AUTO_FOLDER, f"{base}.png_graph.gpickle")

    if not os.path.exists(path_b) or not os.path.exists(path_auto):
        print("file not found")
        print(path_b)
        print(path_auto)
        continue
    try:
        edges_a, nodes_a = load_graph_from_excel(path_a)
    except:
        print(path_a)
    try:
        edges_b, nodes_b = load_graph_from_excel(path_b)
    #auto_graph = nx.read_gpickle(path_auto)
    except:
        print(path_b)
    with open(path_auto, "rb") as f:
        auto_graph = pickle.load(f)

    edges_auto = list(auto_graph.edges())
    nodes_auto = list(auto_graph.nodes())

    compare_and_record(edges_a, nodes_a, edges_b, nodes_b, file, "A_vs_B", results)
    compare_and_record(edges_auto, nodes_auto, edges_a, nodes_a, file, "Auto_vs_A", results)
    compare_and_record(edges_auto, nodes_auto, edges_b, nodes_b, file, "Auto_vs_B", results)

# === Save Outputs ===
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved detailed results to: {OUTPUT_CSV}")

summary = df_out.groupby("comparison")[["precision", "recall", "f1_score", "jaccard_similarity", "netsimile_similarity", "soft_jaccard_similarity", "node_set_similarity", "node_centrality_similarity"]].agg(["mean", "std"])
summary.to_csv(SUMMARY_CSV)
print(f"📊 Saved summary stats to: {SUMMARY_CSV}")
print("\nSummary:\n", summary)
