"""
Final verification of all remaining rating-based numbers so nothing flips later:
  - per-dimension MEAN rating across all ratings -> true ranking (minor 8) + range (minor 7)
  - per-rater / per-score-set totals -> verify the '22.3, 19.3, 19.7' Results claim
  - Fig 4d ranking check
Reads the parsed long ratings (expert_ratings_long.csv).
"""
import pandas as pd
import numpy as np

DIMS = ["Completeness", "Consistency", "Specificity",
        "Plausibility(Nodes)", "Plausibility(Edges)", "Utility/Relevance"]

df = pd.read_csv("expert_ratings_long.csv", dtype={"case": str})

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
