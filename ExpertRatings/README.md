# Expert Ratings

Expert rubric ratings of the LLM-generated causal graphs, added in response to
reviewer request (round-2, comment Q4) to release the expert-rating data.

## File
- `causal_graph_scores_combined.xlsx`

## Sheets
- `scores_rater1`, `scores_rater2`, `scores_rater4`, `scores_rater3` — per-rater scores.
- `scores_compiled`, `scores_summary` — aggregated views.
- `rubric` — the six-dimension rubric (Completeness, Consistency, Specificity, Plausibility of
  Nodes, Plausibility of Edges, Utility/Relevance), each anchored on a 1–5 scale.

## Missing cases
Two ground-truth graphs are unavailable for one annotator each, so the human denominators are
45 graphs per annotator (90 total) versus 46 LLM graphs:
- Group 1, Annotator A: case **195** missing.
- Group 2, Annotator B: case **175** missing.
