"""Exact per-group contrast stats (mean, 95% CI, t, p, Holm-adjusted, dz) in the
user's notation, for rewriting the manuscript paragraph."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("similarity_per_case.csv", dtype={"case": str})
df = df[df.directed == False]
names = {1: {"A": "avani", "B": "minoti"}, 2: {"A": "bhavyaa", "B": "saniya"}}
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
        rows.append(dict(group=group, contrast=name, n=n, md=md,
                         lo=md - tc * se, hi=md + tc * se, t=t, df=n - 1,
                         p=p, dz=md / sd))

# Holm across all 6
ps = np.array([r["p"] for r in rows])
order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
for rank, idx in enumerate(order):
    run = max(run, (m - rank) * ps[idx]); adj[idx] = min(run, 1.0)
for r, a in zip(rows, adj):
    r["p_holm"] = a

for r in rows:
    print(f"G{r['group']} {r['contrast']:22s} md={r['md']:+.3f} CI[{r['lo']:+.3f},{r['hi']:+.3f}] "
          f"t({r['df']})={r['t']:+.2f} P={r['p']:.3f} Padj={r['p_holm']:.3f} dz={r['dz']:+.2f}")
