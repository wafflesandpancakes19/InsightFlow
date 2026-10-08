"""
Topology, distance, and community metrics computed on the
PATIENT-symmetrised graphs (LLM as-is; PATIENT added to human graphs).

Replicates the original methods:
  - topology: diameter & avg shortest path on the largest connected component
    (undirected projection), density (directed), transitivity, avg local
    clustering, triangles.
  - distances: KL divergence (scipy.entropy w/ eps) and EMD (1-D Wasserstein,
    equivalent to the OT euclidean EMD in 1-D) between degree distributions.
  - communities: Leiden (ModularityVertexPartition), Girvan-Newman (k=2),
    label propagation, and Infomap -> number of communities per graph.

Outputs (written to results/): fig4_topology_by_source.csv,
         fig4_communities_by_source.csv, fig4_distances_by_comparison.csv,
         distances_per_case.csv
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy, wasserstein_distance
import igraph as ig
import leidenalg as la
from infomap import Infomap
from networkx.algorithms.community import girvan_newman, label_propagation_communities

from graph_loading import GROUPS, group_ids, load_llm_graph, build_human_graph

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

INFOMAP_SEED = 42  # Infomap is stochastic; fix for reproducibility (matches Leiden seed).


# ---------------- topology ----------------
def topo(G):
    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return dict(diameter=np.nan, avg_path=np.nan, density=np.nan,
                    transitivity=np.nan, clustering=np.nan, triangles=np.nan)
    GC = U.subgraph(max(nx.connected_components(U), key=len))
    try:
        diameter = nx.diameter(GC); avg_path = nx.average_shortest_path_length(GC)
    except Exception:
        diameter = np.nan; avg_path = np.nan
    tri = sum(nx.triangles(U).values()) / 3.0
    return dict(diameter=diameter, avg_path=avg_path, density=nx.density(G),
                transitivity=nx.transitivity(U), clustering=nx.average_clustering(U),
                triangles=tri)


# ---------------- distances ----------------
def degree_dist(G, max_deg):
    degs = [d for _, d in G.degree()]
    hist = np.zeros(max_deg + 1)
    for d in degs:
        hist[d] += 1
    return hist / (hist.sum() if hist.sum() > 0 else 1)

def kl_emd(Ga, Gb):
    md = max(max(dict(Ga.degree()).values(), default=0),
             max(dict(Gb.degree()).values(), default=0))
    p, q = degree_dist(Ga, md), degree_dist(Gb, md)
    kl = float(entropy(p + 1e-10, q + 1e-10))
    emd = float(wasserstein_distance(np.arange(len(p)), np.arange(len(q)), p, q))
    return kl, emd


# ---------------- communities ----------------
def n_leiden(G):
    if G.number_of_nodes() == 0:
        return np.nan
    part = la.find_partition(ig.Graph.from_networkx(G), la.ModularityVertexPartition, seed=42)
    return len([c for c in part])

def n_girvan(G, k=2):
    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return np.nan
    gen = girvan_newman(U)
    try:
        for _ in range(k - 1):
            comms = next(gen)
        comms = next(gen)
    except StopIteration:
        comms = tuple([set(U.nodes())])
    return len(comms)

def n_labelprop(G):
    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return np.nan
    return len(list(label_propagation_communities(U)))

def n_infomap(G):
    """Number of top-level Infomap modules on the undirected projection; isolated
    nodes count as singletons (consistent with the other three algorithms)."""
    if G.number_of_nodes() == 0:
        return np.nan
    U = G.to_undirected()
    node_to_id = {node: i for i, node in enumerate(U.nodes())}
    im = Infomap(f"--two-level --silent --seed {INFOMAP_SEED}")
    for nid in node_to_id.values():
        im.add_node(nid)
    for u, v in U.edges():
        if u != v:
            im.add_link(node_to_id[u], node_to_id[v])
    im.run()
    return len(set(im.get_modules().values()))


# ---------------- collect ----------------
def source_graphs():
    """Return dict: source label -> list of graphs (PATIENT-symmetrised)."""
    llm, A, B = [], [], []
    for group in (1, 2):
        fa, fb = GROUPS[group]["A"], GROUPS[group]["B"]
        for c in group_ids(group):
            g = load_llm_graph(group, c)
            if g is not None:
                llm.append(g)
            ga = build_human_graph(fa, c, with_patient=True)
            if ga is not None:
                A.append(ga)
            gb = build_human_graph(fb, c, with_patient=True)
            if gb is not None:
                B.append(gb)
    return {"LLM": llm, "Annotator A": A, "Annotator B": B}


def run():
    srcs = source_graphs()

    # topology + communities per source
    topo_rows, comm_rows = [], []
    for name, graphs in srcs.items():
        T = [topo(g) for g in graphs]
        dfT = pd.DataFrame(T)
        topo_rows.append(dict(source=name, n=len(graphs),
                              **{k: dfT[k].mean() for k in dfT.columns},
                              **{k + "_sd": dfT[k].std() for k in dfT.columns}))
        lei = [n_leiden(g) for g in graphs]
        gn = [n_girvan(g) for g in graphs]
        lp = [n_labelprop(g) for g in graphs]
        im = [n_infomap(g) for g in graphs]
        comm_rows.append(dict(source=name, n=len(graphs),
                              leiden_mean=np.nanmean(lei), leiden_sd=np.nanstd(lei),
                              girvan_mean=np.nanmean(gn), girvan_sd=np.nanstd(gn),
                              labelprop_mean=np.nanmean(lp), labelprop_sd=np.nanstd(lp),
                              infomap_mean=np.nanmean(im), infomap_sd=np.nanstd(im)))
    pd.DataFrame(topo_rows).to_csv(os.path.join(RESULTS, "fig4_topology_by_source.csv"), index=False)
    pd.DataFrame(comm_rows).to_csv(os.path.join(RESULTS, "fig4_communities_by_source.csv"), index=False)

    # distances per comparison (paired by case)
    dist_rows = []
    for group in (1, 2):
        fa, fb = GROUPS[group]["A"], GROUPS[group]["B"]
        for c in group_ids(group):
            g = load_llm_graph(group, c)
            ga = build_human_graph(fa, c, with_patient=True)
            gb = build_human_graph(fb, c, with_patient=True)
            for comp, (x, y) in {"Auto vs A": (g, ga), "Auto vs B": (g, gb), "A vs B": (ga, gb)}.items():
                if x is not None and y is not None:
                    kl, emd = kl_emd(x, y)
                    dist_rows.append(dict(group=group, case=c, comparison=comp, kl=kl, emd=emd))
    dd = pd.DataFrame(dist_rows)
    dd.to_csv(os.path.join(RESULTS, "distances_per_case.csv"), index=False)
    dsum = dd.groupby("comparison").agg(n=("kl", "size"), kl_mean=("kl", "mean"),
                                        kl_sd=("kl", "std"), emd_mean=("emd", "mean"),
                                        emd_sd=("emd", "std")).reset_index()
    dsum.to_csv(os.path.join(RESULTS, "fig4_distances_by_comparison.csv"), index=False)

    # print
    print("===== TOPOLOGY (PATIENT symmetrised, mean per source) =====")
    dfT = pd.DataFrame(topo_rows)
    for _, r in dfT.iterrows():
        print(f"{r['source']:12s} n={int(r['n']):2d}  diam={r['diameter']:.2f}  path={r['avg_path']:.2f}  "
              f"density={r['density']:.3f}  transit={r['transitivity']:.3f}  clust={r['clustering']:.3f}  tri={r['triangles']:.2f}")
    print("\n===== COMMUNITIES (mean #communities per graph) =====")
    for r in comm_rows:
        print(f"{r['source']:12s}  Leiden={r['leiden_mean']:.2f}  Girvan-Newman={r['girvan_mean']:.2f}  "
              f"LabelProp={r['labelprop_mean']:.2f}  Infomap={r['infomap_mean']:.2f}")
    print("\n===== DISTANCES (degree-distribution, mean per comparison) =====")
    for _, r in dsum.iterrows():
        print(f"{r['comparison']:10s} n={int(r['n']):2d}  KL={r['kl_mean']:.3f}  EMD={r['emd_mean']:.3f}")
    print("\nSaved fig4_topology_by_source.csv, fig4_communities_by_source.csv, fig4_distances_by_comparison.csv")


if __name__ == "__main__":
    run()
