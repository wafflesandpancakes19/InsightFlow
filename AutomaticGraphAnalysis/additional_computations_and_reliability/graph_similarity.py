"""
Faithful recomputation of ALL Table 3 similarity metrics on the
PATIENT-symmetrised graphs, replicating inter_rater_final_new.py exactly, with the
two construction fixes:
  (1) PATIENT node added to the HUMAN graphs (LLM already has it) -> equivalent construction
  (2) both UNDIRECTED (as published) and DIRECTED (direction-sensitive) variants

Metrics per comparison: NetSimile, Mean Edge similarity (soft-Jaccard, SBERT),
Node-set similarity (SBERT centroid cosine), Node-centrality similarity (SBERT+degree).
Node matching + edge normalisation replicate the original pipeline (threshold 0.6).

Outputs: similarity_per_case.csv, similarity_summary.csv
"""
import os, glob, pickle, re
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer, util
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from netsimile import netsimile_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM_THRESHOLD = 0.6
model = SentenceTransformer("all-MiniLM-L6-v2")

GROUPS = {1: {"A": "avani", "B": "minoti"}, 2: {"A": "bhavyaa", "B": "saniya"}}


# ---------------- graph loading (edges + nodes) ----------------
def case_id(f):
    m = re.match(r"(\d+)_out", os.path.basename(f)); return m.group(1) if m else None

def group_ids(g):
    return sorted({case_id(f) for f in glob.glob(os.path.join(ROOT, f"gpickles_{g}", "*.gpickle"))}, key=int)

def load_llm(group, cid):
    p = os.path.join(ROOT, f"gpickles_{group}", f"{cid}_out.png_graph.gpickle")
    if not os.path.exists(p): return None, None
    with open(p, "rb") as f: G = pickle.load(f)
    return list(G.edges()), list(G.nodes())

def load_human(folder, cid):
    p = os.path.join(ROOT, folder, f"{cid}_out.xlsx")
    if not os.path.exists(p): return None, None
    df = pd.read_excel(p)
    node_map = {str(k).strip(): str(v).strip() for k, v in zip(df["Node Code"], df["Nodes"])
                if pd.notna(k) and pd.notna(v)}
    edges = []
    for _, row in df.iterrows():
        s, t = row.get("Edges 1"), row.get("Edges 2")
        if pd.notna(s) and pd.notna(t):
            ss, tt = node_map.get(str(s).strip()), node_map.get(str(t).strip())
            if ss and tt: edges.append((ss, tt))
    return edges, list(node_map.values())

def add_patient(edges, nodes):
    """Replicate make_graph: connect every node with no outgoing edge to PATIENT sink."""
    alln = set(nodes) | {n for e in edges for n in e}
    outgoing = {u for u, v in edges}
    leaves = [n for n in alln if n not in outgoing and n != "PATIENT"]
    new_edges = list(edges) + [(l, "PATIENT") for l in leaves]
    new_nodes = list(alln) + ["PATIENT"]
    return new_edges, new_nodes


# ---------------- original pipeline functions ----------------
def compute_node_matches(nodes_a, nodes_b, threshold=0.6):
    emb_a = model.encode(nodes_a, convert_to_tensor=True)
    emb_b = model.encode(nodes_b, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_a, emb_b)
    matched, used_b = {}, set()
    for i, row in enumerate(sim):
        j = torch.argmax(row).item()
        if row[j] >= threshold and j not in used_b:
            matched[nodes_a[i]] = nodes_b[j]; used_b.add(j)
        else:
            matched[nodes_a[i]] = nodes_a[i]
    for b in nodes_b:
        if b not in matched.values(): matched[b] = b
    return matched

def normalize_edges(edges, mapping, directed):
    out = set()
    for s, t in edges:
        s2, t2 = mapping.get(s, s), mapping.get(t, t)
        out.add((s2, t2) if directed else tuple(sorted((s2, t2))))
    return out

def build_graph(edges, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges); return G

def netsim(edges_a, edges_b, directed):
    Ga, Gb = build_graph(edges_a, directed), build_graph(edges_b, directed)
    try:
        return 1.0 / (1.0 + euclidean(netsimile_features(Ga), netsimile_features(Gb)))
    except Exception as e:
        print("netsim err", e); return np.nan

def soft_jaccard(edges_a, edges_b):
    if not edges_a or not edges_b: return 0.0
    ta = [" ".join(e) for e in edges_a]; tb = [" ".join(e) for e in edges_b]
    ea = model.encode(ta, convert_to_tensor=True); eb = model.encode(tb, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(ea, eb).cpu().numpy()
    matches = [np.max(sims[i]) for i in range(len(ta)) if np.max(sims[i]) >= SIM_THRESHOLD]
    return float(np.mean(matches)) if matches else 0.0

def node_set_sim(nodes_a, nodes_b):
    if not nodes_a or not nodes_b: return 0.0
    ea = model.encode(nodes_a, convert_to_tensor=True); eb = model.encode(nodes_b, convert_to_tensor=True)
    return util.pytorch_cos_sim(torch.mean(ea, 0), torch.mean(eb, 0)).item()

def node_centrality_sim(edges_a, edges_b, directed):
    Ga, Gb = build_graph(edges_a, directed), build_graph(edges_b, directed)
    if not Ga.nodes() or not Gb.nodes(): return 0.0
    ca, cb = nx.degree_centrality(Ga), nx.degree_centrality(Gb)
    ea = model.encode(list(Ga.nodes()), convert_to_tensor=True)
    eb = model.encode(list(Gb.nodes()), convert_to_tensor=True)
    sims = util.pytorch_cos_sim(ea, eb).cpu().numpy()
    bl = list(Gb.nodes()); scores = []
    for i, na in enumerate(Ga.nodes()):
        j = int(np.argmax(sims[i]))
        scores.append(sims[i, j] * ca[na] * cb[bl[j]])
    return float(np.sum(scores) / (np.sum(list(ca.values())) + 1e-6)) if scores else 0.0


def compare(edges1, nodes1, edges2, nodes2, directed):
    m12 = compute_node_matches(nodes1, nodes2, SIM_THRESHOLD)
    m21 = compute_node_matches(nodes2, nodes1, SIM_THRESHOLD)
    merged = {**m12, **{v: k for k, v in m21.items()}}
    n1 = normalize_edges(edges1, merged, directed)
    n2 = normalize_edges(edges2, merged, directed)
    return {
        "netsimile": netsim(n1, n2, directed),
        "mean_edge": soft_jaccard(list(n1), list(n2)),
        "node_set": node_set_sim(nodes1, nodes2),
        "node_centrality": node_centrality_sim(list(n1), list(n2), directed),
    }


def run():
    rows = []
    for group in (1, 2):
        fa, fb = GROUPS[group]["A"], GROUPS[group]["B"]
        for c in group_ids(group):
            le, ln = load_llm(group, c)
            ae, an = load_human(fa, c)
            be, bn = load_human(fb, c)
            if ae is not None: ae, an = add_patient(ae, an)
            if be is not None: be, bn = add_patient(be, bn)
            for directed in (False, True):
                if le is not None and ae is not None:
                    r = compare(le, ln, ae, an, directed); r.update(group=group, case=c, comparison="Auto vs A", directed=directed); rows.append(r)
                if le is not None and be is not None:
                    r = compare(le, ln, be, bn, directed); r.update(group=group, case=c, comparison="Auto vs B", directed=directed); rows.append(r)
                if ae is not None and be is not None:
                    r = compare(ae, an, be, bn, directed); r.update(group=group, case=c, comparison="A vs B", directed=directed); rows.append(r)
        print(f"  group {group} done")

    df = pd.DataFrame(rows)
    df.to_csv("similarity_per_case.csv", index=False)

    metrics = ["netsimile", "mean_edge", "node_set", "node_centrality"]
    agg = []
    for directed in (False, True):
        for group in (1, 2, "TOTAL"):
            for comp in ("A vs B", "Auto vs A", "Auto vs B"):
                sub = df[(df.comparison == comp) & (df.directed == directed)]
                if group != "TOTAL":
                    sub = sub[sub.group == group]
                row = {"decoding": "directed" if directed else "undirected", "group": group, "comparison": comp, "n": len(sub)}
                for m in metrics:
                    row[m + "_mean"] = sub[m].mean(); row[m + "_sd"] = sub[m].std()
                agg.append(row)
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv("similarity_summary.csv", index=False)

    print("\n===== SIMILARITY (PATIENT symmetrised, node-matched pipeline) =====")
    for directed in ("undirected", "directed"):
        print(f"\n--- {directed.upper()} (TOTAL) ---")
        sub = aggdf[(aggdf.decoding == directed) & (aggdf.group == "TOTAL")]
        print(f"{'comparison':10s} n  {'NetSimile':>10s} {'MeanEdge':>10s} {'NodeSet':>10s} {'NodeCentr':>10s}")
        for _, r in sub.iterrows():
            print(f"{r['comparison']:10s} {int(r['n']):2d} {r['netsimile_mean']:10.4f} {r['mean_edge_mean']:10.4f} {r['node_set_mean']:10.4f} {r['node_centrality_mean']:10.4f}")
    print("\nSaved similarity_per_case.csv, similarity_summary.csv")


if __name__ == "__main__":
    run()
