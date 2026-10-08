# Expert Ratings

Expert rubric ratings of the LLM-generated causal graphs, added in response to
reviewer request (round-2, comment Q4) to release the expert-rating data.

## File
- `causal_graph_scores_combined.xlsx`

## Rating design (four raters, incomplete/rotating design)
Each graph received **three ratings**, but the third rater rotated by group:

| Rater | Graphs rated | Also a graph annotator? |
|---|---|---|
| Rater 1 | all 46 | Yes (Group 1 Annotator B) |
| Rater 2 | all 46 | No — independent of annotation |
| Rater 3 | 23 (Group 1) | Yes (Group 1 Annotator A) |
| Rater 4 | 23 (Group 2) | Yes (Group 2 Annotator A) |

- Group 1 graphs were rated by {Rater 3, Rater 1, Rater 2}; Group 2 graphs by {Rater 4, Rater 1, Rater 2}.
- This is an **incomplete (not fully crossed)** design; reliability is therefore reported with
  ordinal Krippendorff's α (valid under missingness) and, for the fully crossed two-rater core
  (Rater 1, Rater 2), an intraclass correlation. See `AutomaticGraphAnalysis/`.
- **Rater/annotator overlap:** three of the four raters also contributed graph annotations; one
  rater (Rater 2) was independent of annotation. Raters scored only the LLM-generated graphs, never
  their own annotations.

## Sheets
- `scores_rater1`, `scores_rater2`, `scores_rater4`, `scores_rater3` — per-rater scores.
- `scores_compiled`, `scores_summary` — aggregated views.
- `rubric` — the six-dimension rubric (Completeness, Consistency, Specificity, Plausibility of
  Nodes, Plausibility of Edges, Utility/Relevance), each anchored on a 1–5 scale.

## Missing cases
Two ground-truth graphs are unavailable for one annotator each, so the human denominators are
45 graphs per annotator (90 total) versus 46 LLM graphs:
- Group 1, Annotator A (Rater 3): case **195** missing.
- Group 2, Annotator B: case **175** missing.
