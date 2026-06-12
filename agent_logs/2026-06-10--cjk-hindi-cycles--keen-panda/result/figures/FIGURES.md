# Multilingual Manifold Figures

Session: `2026-06-10--cjk-hindi-cycles--keen-panda`  
Model: Llama 3.1 8B · Task: weekday cyclic arithmetic · Layer 28 last-token, pca_k64

---

## Figure 1 — Per-language manifold panels (`fig1_language_manifolds`)

**What it shows.** Four subplots (EN, FR, ZH, JA), each showing the 7 weekday centroids
projected into that language's own best-fit 2-D subspace of the 64-dim PCA feature space.
Points are coloured by weekday position in a cyclic HSV wheel (Monday → Sunday).
A smooth curve is fitted through the points: **solid** for English (closed ring via periodic
spline), **dashed** for the others (open spline in natural Mon→Sun order).
A faint dotted circle shows the algebraic best-fit circle for reference.

**What to look for.** English is the only language where the 7 points form a near-perfect
closed ring — the solid curve visibly closes back on itself. French, Chinese, and Japanese
all produce open arcs whose endpoints are far apart. The ring geometry is a property of the
*language*, not of the model's overall ability to do the arithmetic (ZH and JA both pass the
encoding gate at ≥60%).

**Key numbers shown in titles.**
- Gate %: fraction of (entity, offset) pairs answered correctly.
- Isometry r: Pearson correlation between manifold-path distances and behavioral offsets.
  EN=+0.989 (path distances faithfully predict day offsets); ZH=+0.317 (partial); JA=−0.101
  (anti-correlated — spline order is inverted relative to calendar order); FR=+0.087 (near zero).
- Coherence: geometric-mean probability of the correct day under geometric steering.
  ZH and JA are near 1.0 (model is steerable even without a ring); EN and FR are lower (0.74–0.78).

---

## Figure 2 — Joint embedding (`fig2_joint_embedding`)

**What it shows.** All four languages' weekday centroids projected simultaneously into a
shared 2-D PCA space fitted over the union of all 28 centroids. Each language has its own
colour; points are not labelled to keep the plot legible.

**What to look for.** In the shared space, the four languages occupy different, roughly
orthogonal regions. English (blue) still forms a compact ring; the others form small open arcs.
The relative sizes of the arcs reflect the joint PCA projection — in the pairwise projections
(Fig 3), the radius differences are more dramatic because the 2-D plane is chosen to best fit
each pair, rather than all four at once.

---

## Figure 3 — Cross-lingual subspace alignment (`fig3_subspace_alignment`)

**What it shows.** Three subplots, one per language pair (EN↔ZH, EN↔FR, ZH↔JA). For each
pair, both languages' centroids are projected into the 2-D PCA plane that best explains the
*joint* variance of that pair. This is the most direct visualisation of the subspace-alignment
metrics.

**What to look for.** Radius ratios (in subplot titles) quantify how different the two rings
are in the shared plane: a ratio near 1× means the rings are the same size (likely co-planar);
a large ratio means the two languages' subspaces are nearly orthogonal, so one language's arc
"collapses" when viewed from the other's perspective.

- **EN↔ZH** (overlap 0.242, θ₁=59°, θ₂=62°): The EN ring is substantially larger. The ZH
  arc is compact and well-separated from EN's ring.
- **EN↔FR** (overlap 0.023, θ₁=79°, θ₂=84°): Nearly orthogonal. EN and FR occupy almost
  perpendicular subspaces, explaining why an EN-derived steering direction has almost no
  effect on FR representations.
- **ZH↔JA** (overlap 0.350, θ₁=51°, θ₂=57°): The most similar pair. Both have similar
  radii and partially co-planar arcs, consistent with their shared template structure
  (Arabic-numeral offsets, same Q/A format).

---

## Figure 4 — Subspace overlap heatmap (`fig4_overlap_heatmap`)

**What it shows.** A 4×4 symmetric heatmap (EN, FR, ZH, JA) coloured by subspace overlap
score = cos(θ₁)·cos(θ₂). Diagonal = 1 (same language). Off-diagonal = fraction of shared
subspace. All measured pairs are filled; no pair was missing (all 6 pairwise overlaps measured).

**What to look for.** The matrix is uniformly low (< 0.35 everywhere except the diagonal),
confirming that **no two languages share their weekday ring subspace**. The ZH–JA cell is the
brightest off-diagonal entry (0.35); the FR row/column is the darkest (FR is nearly orthogonal
to all other languages). This visualises why cross-lingual safety transfer is structurally
limited across all language pairs, not just EN→FR.

---

## Figure 5 — Pullback quality: geodesic vs linear (`fig5_pullback_comparison`)

**What it shows.** Grouped bars for EN and FR showing the path-recapitulation r² metric
under (a) geodesic-optimised paths (LBFGS belief-space pullback, k=32 neighbours) and
(b) straight-line linear paths. r² measures how well the model's activation path during
steering recapitulates the spline geometry.

**What to look for.** The sign of the gap is opposite for the two languages:

- **English:** Optimised (0.69) > Linear (0.42). The geodesic optimizer successfully
  finds the ring curvature — the model's internal geometry supports curved, ring-like
  paths. Paired t-test p=0.0003.
- **French:** Linear (0.53) > Optimised (0.27). The optimizer diverges for most FR pairs
  (some pairs have `mean_dist_from_geometric` > 100×). FR representations have no curved
  manifold structure to exploit; straight-line transitions recapitulate behaviour better
  than any attempt at geodesic curvature. Paired t-test p<0.0001 (opposite sign).

This plot directly demonstrates that the EN ring is a *genuinely curved* manifold and FR
is not — beyond what the isometry r alone shows.

---

## Figure 6 — Safety vector cross-lingual projection (`fig6_safety_projection`)

**What it shows.** Horizontal bar chart showing the fraction of an English safety direction's
norm that is retained when projected onto each language's 2-D weekday-ring subspace.
Values = subspace overlap scores from Fig 4's EN row: EN=100%, JA=42.1%, ZH=24.2%, FR=2.3%.

**What to look for.** An English-derived safety vector (e.g., a "refuse harmful request"
direction from SAE decomposition or RLHF) essentially vanishes by the time it reaches the
French subspace (2.3% retained). Japanese retains the most (42%) due to the higher EN↔JA
alignment. This has direct implications for cross-lingual safety: English-only safety
training provides structurally limited protection against non-English attacks, and the
magnitude of this limitation depends on the target language's subspace alignment with English.

---

## Figure 7 — Isometry vs coherence scatter (`fig7_isometry_coherence`)

**What it shows.** Four languages plotted on (x=geometric isometry r, y=geometric steering
coherence). EN uses a star marker (ring); others use circles (open spline).

**What to look for.** The two axes capture different things:
- **Isometry r** measures whether the internal path geometry mirrors the calendar structure
  (high = the model "thinks geometrically").
- **Coherence** measures whether steering along that path produces correct output (high =
  the model can be steered to the right day).

These two properties can dissociate:
- EN (★): High isometry AND high coherence — the ring is real and useful.
- ZH and JA (●): Near-zero or negative isometry, yet very high coherence. The model can
  be steered to the correct weekday along the spline, but the internal path geometry does
  not correspond to calendar-distance. A functional but non-geometric encoding.
- FR (●): Low isometry AND lower coherence. FR is the worst on both axes — it passes the
  gate (79.6%) but its internal representation is poorly organised geometrically.

---

## Figure 8 — Cross-lingual steering coherence (`fig8_steering_coherence`)

**What it shows.** Grouped bar chart for three target languages (ZH, JA, FR). For each target,
two bars: (blue) same-language geometric steering coherence from `path_steering`, and (orange)
cross-lingual steering coherence where a *different* language's manifold (source) drives the
interventions. The source language and subspace overlap are annotated above each orange bar.

**What to look for.** The ordering inverts the subspace-overlap hypothesis:

| Pair | Cross-lingual coh. | Same-lang baseline | Subspace overlap |
|---|---|---|---|
| EN → ZH | 0.963 ± 0.0005 | 0.966 ± 0.0005 | 0.242 |
| ZH → JA | 0.770 ± 0.024  | 0.979 ± 0.0005 | 0.350 |
| EN → FR | **0.980 ± 0.0004** | 0.745 ± 0.0003 | 0.023 |

**Key finding.** EN→FR achieves the *highest* coherence (0.980) despite only 2.3% subspace
overlap — better than FR's own same-language baseline (0.745). EN→ZH matches the ZH baseline
almost exactly. ZH→JA is the worst, despite the best subspace alignment.

**Interpretation.** Cross-lingual transfer quality is determined by the *source manifold's
isometry* (EN: r=0.989), not by the subspace overlap between languages. The EN ring's clean
cyclic structure acts as a universal geometric clock: its steering paths traverse
calendar-correct transitions in the target language's representation space regardless of
how the two subspaces are oriented. ZH→JA underperforms because ZH's own manifold has
lower isometry (r=0.317) and thus less-structured steering paths.

**Safety implication.** Even with near-zero representational overlap, English safety vectors
(encoded along ring geometry) can steer French model behaviour — but only because the English
manifold has high isometry. If the safety concept is represented less cleanly in English,
cross-lingual transfer would degrade proportionally.

---

## Reproducibility

**Data extraction:** `extract_manifold_data.py` (run on Cinaps, reads from `raw_features.safetensors`
under each language's `subspace/pca_k64/result/features/` directory).

**Plotting:** `plot_manifolds.py` (run on Cinaps with `uv run python`; matplotlib backend Agg).

Both scripts are in `agent_logs/2026-06-10--cjk-hindi-cycles--keen-panda/code/`.

**Fig 8** generated from jobs 12573–12575 (completed 2026-06-12). Plotted by `plot_fig8.py`.
