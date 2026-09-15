"""
Shared graph-loading and structural helpers for the recomputed analyses.

Addresses two construction issues:
  1. PATIENT-node asymmetry: LLM gpickles already contain a PATIENT hub node
     (leaf -> PATIENT edges added by make_graph). Human graphs (Excel) do not.
     We DO NOT modify the LLM graphs; instead we add the PATIENT node to the
     human graphs using the identical rule, so both sides are constructed
     equivalently.
  2. NetSimile direction: the original pipeline used DIRECTED_GRAPH=False
     (undirected, direction-blind). Here we recompute NetSimile BOTH undirected
     (as published) AND directed (direction-sensitive).

No LLM re-generation, no re-annotation. Structural only (no SBERT needed).

Group 1 = avani (Annotator A) + minoti (Annotator B), LLM = gpickles_1
Group 2 = bhavyaa (Annotator A) + saniya (Annotator B), LLM = gpickles_2
"""

import os
import glob
import pickle
import re
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import euclidean

import sys
# import the repo's netsimile implementation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from netsimile import netsimile_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(__file__)


# ----------------------------------------------------------------------
# Graph loading
# ----------------------------------------------------------------------
def case_id(fname):
    m = re.match(r"(\d+)_out", os.path.basename(fname))
    return m.group(1) if m else None


def load_llm_graph(group, cid):
    """Load an LLM graph (already directed, already contains PATIENT)."""
    folder = os.path.join(ROOT, f"gpickles_{group}")
    path = os.path.join(folder, f"{cid}_out.png_graph.gpickle")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_human_graph_edges(folder, cid):
    """Load a human graph's edges + node list from the annotator Excel file."""
    path = os.path.join(ROOT, folder, f"{cid}_out.xlsx")
    if not os.path.exists(path):
        return None, None
    df = pd.read_excel(path)
    node_map = {
        str(k).strip(): str(v).strip()
        for k, v in zip(df["Node Code"], df["Nodes"])
        if pd.notna(k) and pd.notna(v)
    }
    edges = []
    for _, row in df.iterrows():
        src, tgt = row.get("Edges 1"), row.get("Edges 2")
        if pd.notna(src) and pd.notna(tgt):
            s = node_map.get(str(src).strip())
            t = node_map.get(str(tgt).strip())
            if s and t:
                edges.append((s, t))
    return edges, list(node_map.values())


def add_patient(G):
    """Replicate make_graph(): connect every node with no outgoing edge to a PATIENT sink."""
    G = G.copy()
    all_nodes = set(G.nodes)
    nodes_with_outgoing = {u for u, v in G.edges}
    leaf_nodes = all_nodes - nodes_with_outgoing
    G.add_node("PATIENT")
    for node in leaf_nodes:
        if node != "PATIENT":
            G.add_edge(node, "PATIENT")
    return G


def build_human_graph(folder, cid, with_patient, include_isolated=True):
    edges, nodes = load_human_graph_edges(folder, cid)
    if edges is None:
        return None
    G = nx.DiGraph()
    if include_isolated and nodes:
        G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    if with_patient:
        G = add_patient(G)
    return G


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def descriptives(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = (sum(degs) / n) if n else 0.0
    max_deg = max(degs) if degs else 0
    return n, m, avg_deg, max_deg


def netsimile_sim(Ga, Gb, directed):
    if directed:
        A, B = Ga, Gb
    else:
        A, B = Ga.to_undirected(), Gb.to_undirected()
    try:
        fa = netsimile_features(A)
        fb = netsimile_features(B)
        return 1.0 / (1.0 + euclidean(fa, fb))
    except Exception as e:
        print(f"  NetSimile error: {e}")
        return np.nan


# ----------------------------------------------------------------------
# Group configuration
# ----------------------------------------------------------------------
GROUPS = {
    1: {"A": "avani", "B": "minoti"},
    2: {"A": "bhavyaa", "B": "saniya"},
}


def group_ids(group):
    folder = os.path.join(ROOT, f"gpickles_{group}")
    return sorted({case_id(f) for f in glob.glob(os.path.join(folder, "*.gpickle"))}, key=int)


def summarize(vals):
    v = np.array([x for x in vals if not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, 0)
    return (v.mean(), v.std(), len(v))
