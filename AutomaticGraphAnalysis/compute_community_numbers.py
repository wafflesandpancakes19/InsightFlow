import pandas as pd
import numpy as np
import ast

def parse_communities(cell):
    """
    Convert string representation of list-of-sets into Python list-of-sets.
    """
    if pd.isna(cell):
        return None
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, (list, tuple)):
            return [set(x) for x in parsed]
        return None
    except:
        return None

def community_stats(communities):
    """
    Compute mean and stdev of community sizes.
    """
    if not communities:
        return np.nan, np.nan

    sizes = [len(c) for c in communities]
    mean = float(np.mean(sizes))
    std = float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0
    return mean, std

def process_excel(input_file, output_file):

    df = pd.read_excel(input_file)
    algorithms = [c for c in df.columns if c != "graph_id"]

    # -------------------------------
    # 1. Per-graph stats (wide format)
    # -------------------------------
    per_graph_rows = []

    for _, row in df.iterrows():
        graph_id = row["graph_id"]
        row_dict = {"graph_id": graph_id}

        for algo in algorithms:
            communities = parse_communities(row[algo])
            mean_size, std_size = community_stats(communities)

            row_dict[f"{algo}_mean"] = mean_size
            row_dict[f"{algo}_std"] = std_size

        per_graph_rows.append(row_dict)

    per_graph_df = pd.DataFrame(per_graph_rows)

    # -------------------------------
    # 2. Global stats per algorithm
    # -------------------------------
    global_rows = []

    for algo in algorithms:
        # collect all mean values across graphs
        col = per_graph_df[f"{algo}_mean"].dropna()

        if len(col) == 0:
            global_rows.append({
                "algorithm": algo,
                "global_mean": np.nan,
                "global_std": np.nan,
                "global_median": np.nan,
                "global_min": np.nan,
                "global_max": np.nan
            })
            continue

        global_rows.append({
            "algorithm": algo,
            "global_mean": float(col.mean()),
            "global_std": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
            "global_median": float(col.median()),
            "global_min": float(col.min()),
            "global_max": float(col.max())
        })

    global_df = pd.DataFrame(global_rows)

    # -------------------------------
    # Write single Excel with 2 sheets
    # -------------------------------
    with pd.ExcelWriter(output_file) as writer:
        per_graph_df.to_excel(writer, sheet_name="per_graph_stats", index=False)
        global_df.to_excel(writer, sheet_name="global_stats", index=False)

    print(f"Saved all results to: {output_file}")


# ---------- RUN ----------
process_excel(input_file=r"llm\group_1_auto_communities.xlsx", output_file=r"llm\group_1_auto_community_numbers.xlsx")
process_excel(input_file=r"llm\group_2_auto_communities.xlsx", output_file=r"llm\group_2_auto_community_numbers.xlsx")
process_excel(input_file=r"human\avani_communities.xlsx", output_file=r"human\avani_community_numbers.xlsx")
process_excel(input_file=r"human\bhavyaa_communities.xlsx", output_file=r"human\bhavyaa_community_numbers.xlsx")
process_excel(input_file=r"human\minoti_communities.xlsx", output_file=r"human\minoti_community_numbers.xlsx")
process_excel(input_file=r"human\saniya_communities.xlsx", output_file=r"human\saniya_community_numbers.xlsx")