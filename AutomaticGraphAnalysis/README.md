# Automatic Graph Analysis

The single, corrected analysis pipeline for InsightFlow. It recomputes the reliability,
similarity, and structural results on equivalently constructed graphs:

- the `PATIENT` node is added to **both** the LLM and the human graphs (the LLM gpickles
  already contain it; it is added to the human graphs at load time);
- NetSimile is reported **undirected** (as published) **and directed** (direction-sensitive);
- inter-rater reliability uses the incomplete four-rater design via ordinal Krippendorff's α;
- paired NetSimile tests are run per group with Holm–Bonferroni correction.

## Dependencies
```
pandas  numpy  networkx  scipy
krippendorff                      # ordinal Krippendorff's alpha (reliability)
sentence-transformers             # SBERT semantic similarity (all-MiniLM-L6-v2)
python-igraph  leidenalg          # Leiden community detection
infomap==2.15.1                   # Infomap community detection
```

## Scripts
| Script | Purpose |
|---|---|
| `graph_loading.py` | Shared graph loading; adds the `PATIENT` node to the human graphs so both sides are constructed equivalently |
| `netsimile.py` | NetSimile structural-feature implementation used by the similarity scripts |
| `graph_descriptives.py` | Table 1 descriptives (node/edge/degree counts, with and without `PATIENT`) and NetSimile (undirected & directed) |
| `parse_expert_ratings.py` | Parses the expert-rating workbook and derives the (incomplete) rating design; writes `results/expert_ratings_long.csv` |
| `rater_reliability.py` | Table 2 ordinal **Krippendorff's α** (per dimension + pooled) with bootstrap 95% CIs |
| `rating_statistics.py` | Fig 4d rating means/ranking and the per-dimension rating summary |
| `graph_similarity.py` | Table 3 similarity (NetSimile, Mean-Edge, Node-Set, Node-Centrality), undirected & directed |
| `graph_topology_communities.py` | Fig 4a–c topology, communities (Leiden / Girvan-Newman / label-propagation / **Infomap**), and KL/EMD distances |
| `paired_similarity_tests.py` | Per-group paired NetSimile tests with 95% CIs and Holm–Bonferroni correction |

## Run order
`graph_descriptives.py` -> `parse_expert_ratings.py` -> `rater_reliability.py` ->
`rating_statistics.py` -> `graph_similarity.py` -> `graph_topology_communities.py` ->
`paired_similarity_tests.py`

Each script writes its CSVs into `results/`; later scripts read intermediate files
(`expert_ratings_long.csv`, `table3_similarity_percase.csv`) from there.

## Inputs
Scripts read directly from the repository layout:
- LLM graphs: `LLMGeneratedGraphs/group1/`, `LLMGeneratedGraphs/group2/`
- Human graphs: `AnnotatorGroundTruth/group{1,2}_annotator{1,2}/`
  (A = `annotator1`, B = `annotator2`)
- Expert ratings: `ExpertRatings/causal_graph_scores_combined.xlsx`

## Results
`results/` holds the committed, corrected output CSVs:
`table1_descriptives.csv`, `table2_reliability.csv`, `table3_similarity_summary.csv`,
`table3_similarity_percase.csv`, `fig4_topology_by_source.csv`,
`fig4_communities_by_source.csv`, `fig4_distances_by_comparison.csv`,
`fig4d_rating_summary.csv`, `paired_similarity_tests.csv`.

## Notes
- Four community algorithms are reported — Leiden, Girvan-Newman, label propagation, and
  Infomap — all computed on the **undirected** projection; Leiden and Infomap are seeded at 42.
- Fixed seeds: Krippendorff bootstrap seed = 20260904; Leiden/Infomap seed = 42.
- Two ground-truth graphs are unavailable (Group 1 case **195** for Annotator A;
  Group 2 case **175** for Annotator B), so human denominators are 45 graphs per annotator
  (90 total) versus 46 LLM graphs.
