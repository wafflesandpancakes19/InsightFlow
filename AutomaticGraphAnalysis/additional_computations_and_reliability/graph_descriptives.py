"""
Computes graph descriptives and NetSimile similarity on the equivalently constructed
graphs (the PATIENT node is added to the human graphs so both sides match the LLM graphs).
NetSimile is computed both undirected and directed. Outputs CSVs in this folder.
"""

import numpy as np
import pandas as pd

from graph_loading import (
    GROUPS, group_ids, load_llm_graph, build_human_graph,
    descriptives, netsimile_sim, summarize,
)


# ----------------------------------------------------------------------
# TASK 1 — descriptives (LLM as-is with PATIENT; human with vs without PATIENT)
# ----------------------------------------------------------------------
def compute_descriptives():
    rows = []

    def collect(label, graphs):
        n, m, ad, md = zip(*[descriptives(G) for G in graphs]) if graphs else ([], [], [], [])
        rows.append({
            "source": label, "n_graphs": len(graphs),
            "nodes_mean": np.mean(n), "nodes_sd": np.std(n),
            "edges_mean": np.mean(m), "edges_sd": np.std(m),
            "avgdeg_mean": np.mean(ad), "avgdeg_sd": np.std(ad),
            "maxdeg_mean": np.mean(md), "maxdeg_sd": np.std(md),
        })

    llm_all, humanA_pat, humanB_pat, humanA_nopat, humanB_nopat = [], [], [], [], []

    for group in (1, 2):
        ids = group_ids(group)
        fa, fb = GROUPS[group]["A"], GROUPS[group]["B"]

        llm_g = [load_llm_graph(group, c) for c in ids]
        llm_g = [g for g in llm_g if g is not None]

        hA_pat = [build_human_graph(fa, c, with_patient=True) for c in ids]
        hA_pat = [g for g in hA_pat if g is not None]
        hB_pat = [build_human_graph(fb, c, with_patient=True) for c in ids]
        hB_pat = [g for g in hB_pat if g is not None]
        hA_no = [build_human_graph(fa, c, with_patient=False) for c in ids]
        hA_no = [g for g in hA_no if g is not None]
        hB_no = [build_human_graph(fb, c, with_patient=False) for c in ids]
        hB_no = [g for g in hB_no if g is not None]

        collect(f"LLM G{group} (with PATIENT, as-is)", llm_g)
        collect(f"Annot A G{group} = {fa} (+PATIENT)", hA_pat)
        collect(f"Annot B G{group} = {fb} (+PATIENT)", hB_pat)
        collect(f"Annot A G{group} = {fa} (no PATIENT)", hA_no)
        collect(f"Annot B G{group} = {fb} (no PATIENT)", hB_no)

        llm_all += llm_g
        humanA_pat += hA_pat; humanB_pat += hB_pat
        humanA_nopat += hA_no; humanB_nopat += hB_no

    collect("LLM TOTAL (with PATIENT, as-is)", llm_all)
    collect("Annot A TOTAL (+PATIENT)", humanA_pat)
    collect("Annot B TOTAL (+PATIENT)", humanB_pat)
    collect("Human TOTAL (+PATIENT)", humanA_pat + humanB_pat)
    collect("Annot A TOTAL (no PATIENT)", humanA_nopat)
    collect("Annot B TOTAL (no PATIENT)", humanB_nopat)
    collect("Human TOTAL (no PATIENT)", humanA_nopat + humanB_nopat)

    df = pd.DataFrame(rows)
    df.to_csv("descriptives.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n===== DESCRIPTIVES =====")
    for _, r in df.iterrows():
        print(f"{r['source']:42s} n={int(r['n_graphs']):3d}  "
              f"nodes={r['nodes_mean']:5.2f}+/-{r['nodes_sd']:4.2f}  "
              f"edges={r['edges_mean']:5.2f}+/-{r['edges_sd']:5.2f}  "
              f"avgdeg={r['avgdeg_mean']:4.2f}+/-{r['avgdeg_sd']:4.2f}  "
              f"maxdeg={r['maxdeg_mean']:4.2f}+/-{r['maxdeg_sd']:4.2f}")


# ----------------------------------------------------------------------
# NetSimile, undirected AND directed, PATIENT in both sides
# ----------------------------------------------------------------------
def netsimile_similarity():
    per_case = []

    for group in (1, 2):
        ids = group_ids(group)
        fa, fb = GROUPS[group]["A"], GROUPS[group]["B"]
        for c in ids:
            llm = load_llm_graph(group, c)
            hA = build_human_graph(fa, c, with_patient=True)
            hB = build_human_graph(fb, c, with_patient=True)
            for directed in (False, True):
                if llm is not None and hA is not None:
                    per_case.append([group, c, "Auto vs A", directed,
                                     netsimile_sim(llm, hA, directed)])
                if llm is not None and hB is not None:
                    per_case.append([group, c, "Auto vs B", directed,
                                     netsimile_sim(llm, hB, directed)])
                if hA is not None and hB is not None:
                    per_case.append([group, c, "A vs B", directed,
                                     netsimile_sim(hA, hB, directed)])

    df = pd.DataFrame(per_case, columns=["group", "case", "comparison", "directed", "netsimile"])
    df.to_csv("netsimile_per_case.csv", index=False)

    # Aggregate by group x comparison x directed
    agg_rows = []
    for directed in (False, True):
        for group in (1, 2, "TOTAL"):
            for comp in ("A vs B", "Auto vs A", "Auto vs B"):
                if group == "TOTAL":
                    sub = df[(df.comparison == comp) & (df.directed == directed)]
                else:
                    sub = df[(df.group == group) & (df.comparison == comp) & (df.directed == directed)]
                mean, sd, n = summarize(sub.netsimile.tolist())
                agg_rows.append({
                    "decoding": "directed" if directed else "undirected",
                    "group": group, "comparison": comp,
                    "n": n, "netsimile_mean": mean, "netsimile_sd": sd,
                })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv("netsimile_summary.csv", index=False)

    print("\n===== NETSIMILE (PATIENT in both sides) =====")
    for directed in ("undirected", "directed"):
        print(f"\n--- {directed.upper()} ---")
        sub = agg[agg.decoding == directed]
        for _, r in sub.iterrows():
            g = r["group"]
            print(f"  G{g!s:5s} {r['comparison']:10s} n={int(r['n']):2d}  "
                  f"NetSimile={r['netsimile_mean']:.4f} +/- {r['netsimile_sd']:.4f}")


if __name__ == "__main__":
    compute_descriptives()
    netsimile_similarity()
    print("\nSaved: descriptives.csv, netsimile_per_case.csv, netsimile_summary.csv")
