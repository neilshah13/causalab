# OVERVIEW — behaviour↔activation subspace map & continuous steering

> **FINAL SYNTHESIS → [`CONCLUSIONS.md`](CONCLUSIONS.md)** (2026-06-12) with publication figures
> **F1–F6** under `result/figures/` — head-to-head + significance, two-regime geometry, ring
> skeletons, unprompted/supra-ceiling steering, ablation robustness, where-to-steer.

> **2026-06-11 update (Steps 9–12, jobs 12512–12529).** Four headline additions:
> **(1) Head-to-head:** subspace-inverse steering beats BOTH linear and the paper's spline-manifold
> steering on naturalness — E_BC 0.09 vs 0.45/0.52 (weekdays L23), 2–5× at mid-late layers; ordered
> adjacent-step sweeps (max cyclic jump 1.0 vs 2–3.4 for linear); the paper's manifold>linear gap
> replicates at L19 but vanishes by L31. **Best steering layer is L23/L27, not L31.** 24-token
> generation stays fluent for every method — no post-steering model failure. **(2) Gradient inverse:**
> the Step-7 interior residual (~0.20) was the optimiser budget, not the model — Adam through the
> frozen model reaches any simplex target at median Hellinger ≈0.07 (100% ≤0.2). **(3) Geometry:**
> the valid region has TWO regimes — a calibrated affine core (≤~2 anchor-spacings: chart R²=0.91,
> natural-state isometry rising 0.39→0.77 from L19→L31, the paper's ring as its 7/7-ordered skeleton
> at L23+) inside a saturated shell (still "valid" but chart residual →1.25, entropy collapsed,
> argmax = angular sector). "Everything in the box is valid" was true but blind to this. **(4) Colour
> honesty:** 32/45 colours can never pass the P≥0.5 recovery bar in ANY natural context; among
> reachable colours sparse recovery is 7/10 — plus a curated single-token 12-hue wheel (hues12) that
> steering already sweeps in hue order. Details: REPORT.md Steps 9–12.

**One-line:** for a concept that defines a probability simplex over single tokens (weekdays Δ⁷,
months Δ¹², colours Δ⁴⁵), the model's belief space is a **continuous, low-dimensional, linearly
behaviour-aligned subspace of the late residual stream**, and we can **continuously steer the model
across the gamut by walking that subspace** — validated by deletion-recovery and sparse-anchor tests.

Full narrative + figures: [`REPORT.md`](REPORT.md) (Steps 1–8 + Day-3). All figures rendered to PNG
under `../artifacts/{concept}_simplex/llama31_8b/simplex_coverage/figures/`. Model: Llama-3.1-8B (base),
frame = neutral few-shot, layer 31 unless noted.

## Headline table

| metric | **weekdays Δ⁷** | **months Δ¹²** | **colours Δ⁴⁵** |
|---|---|---|---|
| prompt coverage (retained / total) | 194/225 (86%) | 230/269 (86%) | 127/255 (50%) |
| behaviour simplex dims (Hellinger-PCA 90%) | 6 | 11 | 10 |
| all tokens reachable | yes | yes | **no** (gamut ~10-D, not 45) |
| **map faithfulness** (carrier-sanity, ↓) @L31 | 0.011 | 0.009 | 0.026 |
| **subspace alignment** (mean canonical corr) @L31 | 0.956 | 0.973 | 0.985 |
| onto-ness (coverage-gain vs anchors) | 7.8× | 7.4× | **14.4×** |
| inverse map: vertex / interior Hellinger | 0.03 / 0.20 | 0.10 / 0.20 | — |
| **region recovery** (delete a region, rebuild) | 0.87–1.0 cellRec, maxP≈1.0 | 0.93–1.0 @k13, maxP≈0.97 | Blue 1.0 (weak test: 4 cells, blue-synonym anchors retained); Green 0.74–0.83; **Red 0.35–0.70, non-monotonic in k** |
| **sparse steering** (anchor few → recover unprompted) | keep **1**→6/6; keep 2→5/5 | keep 2→9/10; keep 4→8/8 @k13 | raw: keep 3→13/42 — but **32/45 colours have natural max-P <0.5** (threshold unreachable); among *reachable* held-outs: keep 3→**7/10**, keep 8→3/5 (misses: Navy, Violet — absorbed by Blue/Purple) |
| **continuous steering** fidelity / valid @L31 | 0.095 / 100% | 0.105 / 100% | k32: 0.096 / 100%; **k16–24: 0.11–0.13, valid 82–84%, p90 up to 0.91** |

## The results, in order

1. **Prompt set spans the simplex.** A diverse, non-leaking neutral-few-shot prompt set scatters
   next-token distributions across each simplex (vertices→interior). Days/months retain ~86%; colours
   ~50% (messier gamut). The weekday cloud even recovers the **cyclic ring** in Hellinger-PCA.
2. **The behaviour subspace is linearly embedded in the late residual stream.** It emerges sharply
   ~**L19** and saturates by L31 (canonical corr →0.96–0.99; map faithfulness — behaviour as a function
   of the last-token activation — reaches Hellinger ≈0.01). The behaviour-relevant subspace is a
   ~|Z|−1-dim slice of a larger (~3×) activation subspace.
3. **The valid activation region is continuous & onto.** Perturbing/walking the anchor subspace stays
   valid out to ~2 anchor-spacings and **fills the behavioural simplex** (coverage-gain 7–14×), not just
   smears near anchors.
4. **The map generalises.** Months (Δ¹²) and colours (Δ⁴⁵) reproduce every weekday phenomenon, with the
   **dimensionality law k ≈ |Z|−1** (days need k≳6–8, months k≳13). *Caveat found & fixed:* a months
   recovery run at k=8 was a false negative; at k=13 it recovers (maxP 0.10→0.97).
5. **Validation — leave-a-region-out recovery.** Delete an entire behavioural region's anchors, rebuild
   the map from the rest: days recover 87–100% of the region (≈ random-deletion control); months 93–100%
   at k=13. The subspace spans deleted directions → it is a continuous subspace, not memorised points.
6. **Sparse-anchor steering ("prompt for K, steer all of Z").** Days: anchoring **1** day recovers all 6
   others; months: anchoring 2 recovers 9/10 (at k=13). **Colours: the raw numbers (≈13/42, flat in k)
   are dominated by an unreachable threshold** — `color_ceilings.json` shows 32/45 colours never reach
   P≥0.5 in ANY of the 255 natural prompts (Charcoal/Rose/Plum/Peach ceilings ≈0.01). Among held-out
   colours that are naturally reachable (ceiling ≥0.5), sparse recovery is **7/10 (keep 3) / 3/5 (keep 8)**.
   The residual genuine misses are hue-shades absorbed by their dominant neighbour (Navy→Blue,
   Violet→Purple). CCA caps at |Z|+1, so large gamuts need PCA/diff.
7. **Continuous steering (the payoff — with caveats).** Inverting the map along a target path and
   patching the activations makes the model's output **sweep the gamut in order**: weekdays
   Mon→Tue→…→Sun→Mon, months Jan→…→Dec, colours Red→Orange→Yellow→Green→Blue→Purple (and a clean
   Red→Blue sweep). Days/months: fidelity ≈0.1, 100% valid; **fidelity improves with depth**
   (0.15@L19 → 0.095@L31). Colours need k≈32: at k16–24 validity drops to 82–84% with p90 fidelity
   up to 0.91 (the |Z|−1 law again). Caveats: measured on the FIRST TOKEN only with a single carrier
   prompt, against idealized vertex-geodesic targets, with no baseline comparison — the
   linear-vs-manifold-vs-subspace head-to-head + multi-token generation check is running (jobs
   12513/12516; `steer_compare.py`, `inverse_opt.py`).

## Bottom line

The behaviour↔activation map is real, continuous, low-dimensional, behaviour-aligned at late layers,
**onto the model's natural reachable set**, and **steerable** — and it generalises across simplices of
size 7→12→45. Two limits, now quantified: (1) the **intrinsic dimensionality of the gamut** — colours'
effective dim ~10 ≪ 45; (2) the **model's natural output prior** — 32/45 colours never exceed P=0.5 in
any natural context, so "full gamut recovery" was never on the table; among naturally-reachable colours,
sparse recovery reaches ~70%. What the session has NOT yet shown (in flight, jobs 12512–12516): that
subspace steering beats/matches linear and spline-manifold steering on naturalness (E_BC) and
post-steering fluency, and what metric geometry the map carries inside the valid region.

## Open / next

- **Gemma cross-model**: Gemma tokenizes **94 single-token colours vs Llama's 80** (superset incl.
  Magenta/Indigo/Maroon/…). Capturing on Gemma tests architecture-generality + a richer colour gamut.
- Gradient inverse optimiser to tighten the interior-mixture residual (~0.2) → sharper steering.
- Colour gamut: restrict to its ~10–15 effective colours, or use a perceptual ordering, for cleaner
  sparse recovery.
