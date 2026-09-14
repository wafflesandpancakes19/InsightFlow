# Expert Ratings

Expert rubric ratings of the LLM-generated causal graphs, added in response to
reviewer request (round-2, comment Q4) to release the expert-rating data.

## File
- `causal_graph_scores_combined.xlsx`

## Rating design (four raters, incomplete/rotating design)
Each graph received **three ratings**, but the third rater rotated by group:

| Rater | Graphs rated | Also a graph annotator? |
|---|---|---|
| minoti | all 46 | Yes (Group 1 Annotator B) |
| reena | all 46 | No — independent of annotation |
| avani | 23 (Group 1) | Yes (Group 1 Annotator A) |
| bhavyaa | 23 (Group 2) | Yes (Group 2 Annotator A) |

- Group 1 graphs were rated by {avani, minoti, reena}; Group 2 graphs by {bhavyaa, minoti, reena}.
- This is an **incomplete (not fully crossed)** design; reliability is therefore reported with
  ordinal Krippendorff's α (valid under missingness) and, for the fully crossed two-rater core
  (minoti, reena), an intraclass correlation. See `AutomaticGraphAnalysis/Round2Recompute/`.
- **Rater/annotator overlap:** three of the four raters also contributed graph annotations; one
  rater (reena) was independent of annotation. Raters scored only the LLM-generated graphs, never
  their own annotations.

## Sheets
- `scores_minoti`, `scores_reena`, `scores_bhavyaa`, `scores_avani` — per-rater scores.
- `scores_compiled`, `scores_summary` — aggregated views.
- `rubric` — the six-dimension rubric (Completeness, Consistency, Specificity, Plausibility of
  Nodes, Plausibility of Edges, Utility/Relevance), each anchored on a 1–5 scale.

## Missing cases
Two ground-truth graphs are unavailable for one annotator each, so the human denominators are
45 graphs per annotator (90 total) versus 46 LLM graphs:
- Group 1, Annotator A (avani): case **195** missing.
- Group 2, Annotator B (saniya): case **175** missing.
