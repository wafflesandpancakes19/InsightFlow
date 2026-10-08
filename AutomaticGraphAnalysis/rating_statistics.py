"""
Rating-based summaries from expert_ratings_long.csv:
  - per-dimension mean rating and ranking
  - per-rater / per-score-set total means
  - Fig 4d summary (grand mean + between-score-set SD) -> results/fig4d_rating_summary.csv
"""
import os
import pandas as pd
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

DIMS = ["Completeness", "Consistency", "Specificity",
        "Plausibility(Nodes)", "Plausibility(Edges)", "Utility/Relevance"]

df = pd.read_csv(os.path.join(RESULTS, "expert_ratings_long.csv"), dtype={"case": str})

print("===== PER-DIMENSION MEAN RATING (all raters, all graphs) =====")
means = {}
for d in DIMS:
    v = df[d].dropna()
    means[d] = v.mean()
    print(f"  {d:22s} mean={v.mean():.3f}  sd={v.std():.3f}  n={len(v)}")

ranked = sorted(means.items(), key=lambda x: -x[1])
print("\nRanking (highest -> lowest):")
for d, m in ranked:
    print(f"  {d:22s} {m:.3f}")
print(f"\nRange of means: {min(means.values()):.2f} to {max(means.values()):.2f}")
print(f"Highest dimension: {ranked[0][0]}  | Lowest: {ranked[-1][0]}")

print("\n===== PER-RATER MEAN TOTAL (out of 30) =====")
df["total"] = df[DIMS].sum(axis=1, min_count=len(DIMS))
for r, sub in df.groupby("rater"):
    t = sub["total"].dropna()
    print(f"  {r:10s} mean total={t.mean():.2f}  sd={t.std():.2f}  n={len(t)}")

print("\n===== PER-GRAPH: 3 ratings -> mean of the 3 rater-totals per graph =====")
# reconstruct 'score set' style: for each graph, the totals of its 3 raters
per_graph_totals = df.groupby("case")["total"].apply(lambda s: s.dropna().tolist())
allt = [t for lst in per_graph_totals for t in lst]
print(f"  overall mean total per rating = {np.mean(allt):.2f}, sd = {np.std(allt, ddof=1):.2f}, n_ratings={len(allt)}")

# Fig 4d: per-dimension grand mean + between-score-set SD.
# Score sets: 1 = rater1 (all 46), 2 = rater2 (all 46), 3 = rater3 (G1) + rater4 (G2).
SCORE_SETS = {"set1": ["rater1"], "set2": ["rater2"], "set3": ["rater3", "rater4"]}
fig4d = []
for d in DIMS:
    set_means = [df[df.rater.isin(rs)][d].mean() for rs in SCORE_SETS.values()]
    fig4d.append({"dimension": d, "mean": df[d].mean(), "sd": np.std(set_means, ddof=1)})
pd.DataFrame(fig4d).to_csv(os.path.join(RESULTS, "fig4d_rating_summary.csv"), index=False)
print("\nSaved fig4d_rating_summary.csv")
