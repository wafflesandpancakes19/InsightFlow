"""
Inter-rater reliability for the incomplete (rotating-rater) rating design.

Design (derived from the sheets): minoti and reena rated all 46 graphs; avani rated the 23
Group-1 graphs; bhavyaa rated the 23 Group-2 graphs. Every graph has three ratings, but the
third rater rotates by group. Because this is an incomplete design and the ratings are ordinal,
reliability is reported with ordinal Krippendorff's alpha (valid under missingness and
appropriate for ordinal data), per dimension and pooled, with bootstrap 95% CIs.

Requires expert_ratings_long.csv (produced by parse_expert_ratings.py).
Outputs: rater_reliability_results.csv
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import krippendorff

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIMS = ["Completeness", "Consistency", "Specificity",
        "Plausibility(Nodes)", "Plausibility(Edges)", "Utility/Relevance"]
RATERS = ["minoti", "reena", "avani", "bhavyaa"]
rng = np.random.default_rng(20260904)


def group_ids(group):
    folder = os.path.join(ROOT, f"gpickles_{group}")
    return {re.match(r"(\d+)_out", os.path.basename(f)).group(1)
            for f in glob.glob(os.path.join(folder, "*.gpickle"))}


def load_long():
    df = pd.read_csv("expert_ratings_long.csv", dtype={"case": str})
    g1, g2 = group_ids(1), group_ids(2)
    df["group"] = df["case"].map(lambda c: 1 if c in g1 else (2 if c in g2 else np.nan))
    return df


def kripp_alpha(matrix):
    return krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")


def alpha_with_ci(matrix, n_boot=2000):
    alpha = kripp_alpha(matrix)
    n = matrix.shape[1]
    boots = []
    for _ in range(n_boot):
        cols = rng.integers(0, n, n)
        try:
            a = kripp_alpha(matrix[:, cols])
            if not np.isnan(a):
                boots.append(a)
        except Exception:
            pass
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return alpha, lo, hi


def build_matrix(df, value_col, unit_col, raters):
    units = sorted(df[unit_col].unique())
    idx = {u: i for i, u in enumerate(units)}
    M = np.full((len(raters), len(units)), np.nan)
    for r_i, r in enumerate(raters):
        for _, row in df[df.rater == r].iterrows():
            v = row[value_col]
            if not pd.isna(v):
                M[r_i, idx[row[unit_col]]] = v
    return M


def main():
    df = load_long()
    rows = []
    print("===== Krippendorff's alpha (ordinal, incomplete 4-rater design) =====")
    for dim in DIMS:
        M = build_matrix(df, dim, "case", RATERS)
        a, lo, hi = alpha_with_ci(M)
        rows.append({"dimension": dim, "kripp_alpha_ordinal": a,
                     "kripp_ci_lo": lo, "kripp_ci_hi": hi})
        print(f"{dim:22s} {a:6.3f} [{lo:6.3f}, {hi:6.3f}]")

    # pooled: stack all six dimensions as separate units (case | dimension)
    dlong = df.melt(id_vars=["rater", "case", "group"], value_vars=DIMS,
                    var_name="dim", value_name="score").dropna(subset=["score"])
    dlong["unit"] = dlong["case"] + "|" + dlong["dim"]
    M = build_matrix(dlong, "score", "unit", RATERS)
    a, lo, hi = alpha_with_ci(M)
    rows.append({"dimension": "POOLED", "kripp_alpha_ordinal": a,
                 "kripp_ci_lo": lo, "kripp_ci_hi": hi})
    print(f"{'POOLED':22s} {a:6.3f} [{lo:6.3f}, {hi:6.3f}]")

    pd.DataFrame(rows).to_csv("rater_reliability_results.csv", index=False)
    print("\nSaved rater_reliability_results.csv")


if __name__ == "__main__":
    main()
