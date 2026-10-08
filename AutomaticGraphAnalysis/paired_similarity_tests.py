"""Per-group paired NetSimile tests (undirected), matching the manuscript:
each group is analysed separately (A/B are positional labels that differ by
group), 3 contrasts per group = 6 tests, Holm-Bonferroni across all six.
Reports mean diff, 95% CI, t, df, p, Holm-adjusted p, and Cohen's dz.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS, "table3_similarity_percase.csv"), dtype={"case": str})
df = df[df.directed == False]  # undirected, as in the manuscript
CON = {"sim(A,LLM)-sim(A,B)": ("Auto vs A", "A vs B"),
       "sim(B,LLM)-sim(A,B)": ("Auto vs B", "A vs B"),
       "sim(A,LLM)-sim(B,LLM)": ("Auto vs A", "Auto vs B")}

rows = []
for group in (1, 2):
    sub = df[df.group == group]
    wide = sub.pivot_table(index="case", columns="comparison", values="netsimile")
    for name, (a, b) in CON.items():
        pair = wide[[a, b]].dropna()
        d = (pair[a] - pair[b]).values
        n = len(d); md = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        t, p = stats.ttest_rel(pair[a], pair[b])
        rows.append(dict(group=group, contrast=name, n=n, mean_diff=md,
                         ci_lo=md - tc * se, ci_hi=md + tc * se, t=t, df=n - 1,
                         p_ttest=p, cohen_dz=md / sd))

res = pd.DataFrame(rows)

# Holm-Bonferroni across all six tests
ps = res["p_ttest"].values
m = len(ps); adj = np.empty(m); run = 0.0
for rank, idx in enumerate(np.argsort(ps)):
    run = max(run, (m - rank) * ps[idx]); adj[idx] = min(run, 1.0)
res["p_ttest_holm"] = adj

res = res.sort_values(["group", "contrast"]).reset_index(drop=True)
res.to_csv(os.path.join(RESULTS, "paired_similarity_tests.csv"), index=False)

for _, r in res.iterrows():
    print(f"G{int(r['group'])} {r['contrast']:22s} md={r['mean_diff']:+.3f} "
          f"CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] t({int(r['df'])})={r['t']:+.2f} "
          f"P={r['p_ttest']:.3f} Padj={r['p_ttest_holm']:.3f} dz={r['cohen_dz']:+.2f}")
print("\nSaved paired_similarity_tests.csv")
