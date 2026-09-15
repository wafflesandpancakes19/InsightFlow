# Recomputed Graph Analysis

Statistical/analysis code that recomputes the reliability, similarity, and structural results
using equivalent graph construction (the `PATIENT` node is added to the human graphs so both
sides match the LLM graphs), direction-sensitive metrics, design-appropriate inter-rater
reliability, and transcript-level paired tests.

## Dependencies
```
pandas  numpy  networkx  scipy
krippendorff                      # ordinal Krippendorff's alpha (reliability)
sentence-transformers             # SBERT semantic similarity (all-MiniLM-L6-v2)
python-igraph  leidenalg          # Leiden community detection
```

## Scripts
| Script | Purpose |
|---|---|
| `graph_loading.py` | Shared graph loading; adds the `PATIENT` node to the human graphs so both sides are constructed equivalently |
| `netsimile.py` | NetSimile structural-feature implementation used by the similarity scripts |
| `graph_descriptives.py` | Descriptives (node/edge/degree counts, with and without the `PATIENT` node) and NetSimile (undirected & directed) |
| `parse_expert_ratings.py` | Parses the expert-rating workbook and derives the (incomplete) rating design; writes `expert_ratings_long.csv` |
| `rater_reliability.py` | Ordinal **Krippendorff's α** (per dimension + pooled) with bootstrap 95% CIs |
| `rating_statistics.py` | Per-dimension rating means, ranking, and expert totals |
| `graph_similarity.py` | Table 3 similarity (NetSimile, Mean-Edge, Node-Set, Node-Centrality), undirected & directed; writes per-case similarities |
| `graph_topology_communities.py` | Topology (diameter, path length, density, transitivity, clustering, triangles), communities (Leiden/Girvan-Newman/label-propagation), and KL/EMD distances |
| `paired_similarity_tests.py` | Per-group transcript-level paired NetSimile tests with 95% CIs and Holm–Bonferroni correction |

## Typical run order
`graph_descriptives.py` -> `parse_expert_ratings.py` -> `rater_reliability.py` ->
`rating_statistics.py` -> `graph_similarity.py` -> `graph_topology_communities.py` ->
`paired_similarity_tests.py`
(each writes CSVs that later scripts and the summary rely on).

## Reproducibility notes
- Fixed seeds: Krippendorff bootstrap seed = 20260904; Leiden seed = 42. Reliability is reported
  with ordinal Krippendorff's alpha only (valid under the incomplete rating design).
- Paths in the scripts assume the analysis working-directory layout (e.g. `gpickles_1/`,
  `human_ratings/`); adjust the folder paths to this repository's layout
  (`LLMGeneratedGraphs/`, `AnnotatorGroundTruth/`, `ExpertRatings/`) before running.
