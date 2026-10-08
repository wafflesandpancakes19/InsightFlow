"""
Parse the four rater sheets into a tidy (rater, file, group, dimension) long
table, derive the incomplete rating design, and recompute inter-rater
reliability appropriately for it.
"""
import os
import numpy as np
import pandas as pd

XLSX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "ExpertRatings", "causal_graph_scores_combined.xlsx"))
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

RATER_SHEETS = ["scores_rater1", "scores_rater2", "scores_rater4", "scores_rater3"]
DIMS = ["Completeness", "Consistency", "Specificity",
        "Plausibility(Nodes)", "Plausibility(Edges)", "Utility/Relevance"]


def parse_rater_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    rater = sheet.replace("scores_", "")
    cur_group = None
    out = []
    for _, row in df.iterrows():
        marker = str(row.get("Unnamed: 0")).strip().lower()
        if marker.startswith("group 1"):
            cur_group = 1
        elif marker.startswith("group 2"):
            cur_group = 2
        fname = row.get("File Name")
        if pd.isna(fname):
            continue
        fname = str(fname).strip()
        if not fname.endswith(".csv"):
            continue
        # require at least one numeric score
        scores = {d: row.get(d) for d in DIMS}
        if all(pd.isna(v) for v in scores.values()):
            continue
        cid = fname.replace(".csv", "")
        rec = {"rater": rater, "group": cur_group, "case": cid}
        for d in DIMS:
            rec[d] = pd.to_numeric(scores[d], errors="coerce")
        out.append(rec)
    return pd.DataFrame(out)


def main():
    frames = [parse_rater_sheet(XLSX, s) for s in RATER_SHEETS]
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(os.path.join(RESULTS, "expert_ratings_long.csv"), index=False)

    print("===== COVERAGE: how many graphs each rater scored, by group =====")
    cov = long.groupby(["rater", "group"])["case"].nunique().unstack(fill_value=0)
    print(cov)
    print("\nTotal graphs scored per rater:")
    print(long.groupby("rater")["case"].nunique())

    # ratings per (group, case): how many raters scored each graph
    print("\n===== RATINGS PER GRAPH (should show design) =====")
    rc = long.groupby(["group", "case"])["rater"].nunique()
    print("Distribution of #raters per graph:")
    print(rc.value_counts().sort_index())

    # which raters cover which group
    print("\n===== WHICH RATERS RATE WHICH GROUP =====")
    for g in (1, 2):
        raters_g = sorted(long[long.group == g]["rater"].unique())
        ncases = long[long.group == g]["case"].nunique()
        print(f"  Group {g}: raters={raters_g}, #graphs={ncases}")

    # per-graph rater membership sample
    print("\n===== SAMPLE per-graph rater sets =====")
    membership = long.groupby(["group", "case"])["rater"].apply(lambda s: ",".join(sorted(s)))
    print(membership.head(10).to_string())
    print("...")
    print("\nUnique rater-set patterns:")
    print(membership.value_counts())


if __name__ == "__main__":
    main()
