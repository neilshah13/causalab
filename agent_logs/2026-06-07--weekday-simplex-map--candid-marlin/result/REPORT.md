# REPORT — weekday-simplex-map

**Steps:** 1) prompt set + simplex coverage · 2) behaviour-vs-activation subspace (linear) ·
3) mapping the activation subspace by perturbing anchors (causal) · 4) shape of the valid region —
is the map onto? · 5) concept-general routine + months (Δ¹²) generalisation + overnight batch ·
6) leave-a-region-out recovery (method validation) ·
7) inverse map (sharp onto-test) · 8) sparse-anchor steering-completeness ("prompt for K, steer all of Z") ·
**9) steering head-to-head vs paper baselines + generation check · 10) gradient inverse (interior
residual retracted) · 11) geometry of the map (two regimes; ring skeleton; isometry-by-depth) ·
12) honest colour accounting (ceilings; hue wheel).**

**Session:** `agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/`
**Date:** 2026-06-07 · **Model:** Llama-3.1-8B (base) · **Cluster:** cinaps (1× A6000, jobs 12418–12422)

## Objective (recap)

Build a prompt set whose next-token weekday distributions densely and diversely **cover the
behavioural simplex** — the region of Δ⁷ over {Mon…Sun} where weekday mass > 90% and "other"
mass < 10% — as the foundation for later mapping behaviour space (M_y) to activation space (M_h).
Decide the count, generate the prompts, validate coverage on the Llama model, flag issues.

## TL;DR

- **Deliverable met.** A 225-prompt set across 9 families; **194 / 225 (86%) pass the mass filter**
  on **base Llama-3.1-8B** with the chosen prompt frame. The retained points span **6 Hellinger-PCA
  dimensions** (participation ratio **5.85** ≈ the full simplex dimension), with entropy ranging
  0.01 → 1.93 nats (near-vertex → near-uniform), **all 7 days reachable and balanced**, and **39%
  genuinely interior** points. All four success criteria are satisfied.
- **The behavioural simplex recovers the weekday cycle.** In Hellinger-PCA the 194 points trace a
  clean ring Tue→Wed→Thu→Fri→Sat→Sun→Mon→Tue with interior points filling the centre
  (`result/figures/pca_scatter_by_argmax_day.png`).
- **Key methodological finding — frame matters more than the prompts.** Eliciting *spread* belief
  states (not just high mass) required engineering the prompt *frame*; two natural choices fail in
  opposite ways (details below).

## What was built

`code/methods/weekday_prompts/prompts.py` → `prompts.json` (225 prompts). Each prompt is an
**answer-first completion stem** ("…:") so the model's **first token is a day** (per the
requirement). Families and their target simplex regions:

| Family | n | Targets | Example stem |
|---|---|---|---|
| A_open | 12 | interior | `A day of the week:` |
| B_neg1 | 49 | 6-day faces | `A day of the week that is not Monday:` |
| C_neg2 | 21 | 5-day faces | `A day of the week that is neither Monday nor Tuesday:` |
| D_letter (+neg) | 15 | edges {Tue,Thu},{Sat,Sun} / vertices | `A day of the week that starts with the letter T:` |
| E_succ | 42 | vertices + cyclic neighbour spread | `The day after Monday:` |
| F_semantic | 24 | edges / vertices / bimodal | `The first day of the week:` |
| G_ordinal | 20 | convention-dependent spread | `The third day of the week:` |
| H_today/graded | 18 | vertices + soft interior | `Today is Monday; tomorrow:` |
| I_multi | 24 | small faces / edges | `A weekend day that is not Sunday:` |

**Count rationale.** The shipped weekday arithmetic task enumerates 7×7 = 49 vertex-clustered
prompts; subspace PCA there uses k≈8–16 (~2–3× the 7 classes). To span the full 6-D simplex *and*
support the later 4096-D activation PCA we targeted ~225–250; 194 retained is ~4× the old set and
~32× the simplex dimension — comfortable for PCA.

## The frame investigation (the substantive result)

We measure the **first-token** distribution restricted to the 7 weekday tokens (+ "other"). All 7
days are **single tokens** in the Llama-3.1 tokenizer (verified); per-day mass sums leading-space /
capitalisation variants. The challenge was not *which* prompts but *how to frame them* so the first
token is a day **without** distorting the model's genuine belief. Four regimes, on the 24–45-prompt
probe and the full set:

| Regime | pass (other<10%) | interior frac | what happens |
|---|---|---|---|
| base, `Q:\nA:` / bare stem | **0%** | — | argmax is the right day 98% of the time, but ~30–50% mass bleeds to prose lead-ins (` I`, ` The`, ` a`, ` called`). |
| **instruct + "answer in one word"** | **98%** | **0.26** | high mass but **collapses the simplex to its vertices** — "The day after Monday" → Tue=**1.00** (RLHF decisiveness); interior structure destroyed. |
| **base + neutral few-shot, colon stems** ✅ | **86%** | **0.39** | strips the prose scaffolding *without* prescribing a day; preserves graded belief (e.g. "not Monday:" → Tue .43/Wed .20/Thu .08/Fri .06/Sat .02). |
| instruct + "reply with a day" (reference) | 100% | 0.26 | identical to "one word" — naming the answer space buys nothing, so it is rejected as leaky. |

Two decisions came out of this:

1. **Don't constrain content, constrain form.** Instructions that name the answer ("reply with a
   day") *or* a one-word few-shot of only-day exemplars **leak the answer space**. The clean fix is
   a **neutral few-shot** that demonstrates terse *answer-first* completions on **non-weekday**
   tasks (color, fruit, letter, "the letter after A: B") — it induces the format with zero
   answer-space leakage. Empirically it matches the leaky upper bound on mass.
2. **Use the base model, not instruct.** "Answer in one word" on the RLHF instruct model gives 98%
   mass but **collapses every prompt to a vertex** (interior 0.26) — the opposite of spanning the
   simplex. The base model keeps the spread (interior 0.39, and genuinely graded distributions),
   which is exactly what the behaviour-manifold work needs. Base is also the model the
   manifold-steering activation work uses, so downstream M_h pairing stays consistent.

Frame engineering also fixed a stem-shape issue: copular "…is"/"…on" stems made the base model
continue with prose; switching every family to **answer-first colon "…:"** raised the relational/
ordinal families from ~0% to ~100% retained (E 1→42, G 0→20, H_today 0→14) with no leakage.

## Coverage results (base, 194 retained)

`coverage_metrics.json`, figures in `result/figures/`.

- **Dimensional spread:** Hellinger-PCA explained-variance = [0.24, 0.23, 0.15, 0.12, 0.10, 0.10] →
  **6 dims for 90%**, participation ratio **5.85**. Variance is *not* concentrated on one axis: the
  points fill a genuinely ~6-D region (the simplex is 7-D incl. "other").
- **Vertices + interior:** entropy over the 7 days spans **0.01 → 1.93 nats** (max log 7 = 1.95);
  **39% interior** (≥ ½·log 7). So we cover near-deterministic answers *and* near-uniform beliefs
  and everything between.
- **Day balance:** argmax counts Mon 49 / Tue 26 / Wed 31 / Thu 21 / Fri 29 / Sat 19 / Sun 19 — all
  reachable, Monday still the model's mild default but no longer dominant.
- **Pairwise spread:** Hellinger distances p10/median/p90 = 0.39 / 0.74 / 0.95 (min 0.01, max 0.99).
- **Per-family yield:** A 7/12 · B 35/49 · C 21/21 · D 5/10 + 5/5 · E 42/42 · F 21/24 · G 20/20 ·
  H 4/4 + 14/14 · I 20/24. Every family contributes.

### Hypotheses

- **H1 (families land where intended): SUPPORTED.** argmax-in-intended = **97%**; the family
  centroid heatmap shows distinct day-biases (negation→faces, letter→{Tue,Thu}/{Sat,Sun}, etc.).
- **H2 (mass stays on weekdays): SUPPORTED with caveat.** 86% pass other<10% — but only after the
  frame engineering above; the naive frames fail.
- **H3 (cyclic neighbour spread): SUPPORTED.** For "The day after X:", the 2nd-highest day is
  **always cyclically adjacent** (e.g. "day after Tuesday" → Wed 0.70, **Thu 0.26**), reproducing
  the paper's §2.1 hypothesis. The full point cloud forms a ring (see figure).

## Comparison: base vs instruct

| | base + neutral few-shot ✅ | instruct + one-word |
|---|---|---|
| retained | 194 (86%) | 221 (98%) |
| **interior fraction** | **0.39** | 0.26 |
| PCA dims (90%) / PR | 6 / 5.85 | 6 / 5.67 |
| geometry | ring **with filled interior** | ring with **tight vertex clusters** |

Instruct yields more points but hugs the corners; base spans the interior. For the behaviour↔
activation map, base is the right tool.

## Flags / things to improve

1. **194 < the ~200 target (barely).** Trivially fixable: add ~15–20 stems to B/F/I or a second
   paraphrase of A — the builder already dedups, so this is a one-line change if you want ≥200.
2. **Mild Monday bias** in the open/semantic families (A, F) — the model's genuine prior. If we want
   denser Thu/Sat/Sun *interior* coverage, add targeted multi-constraint (family I) stems favouring
   late-week subsets.
3. **D_letter dropped to 5/10** — the "beginning with {L}" paraphrase underperforms "starts with the
   letter {L}". Worth pruning the weak paraphrase.
4. **Frame is a confound for the eventual M_h step.** The neutral few-shot prefix is part of the
   prompt, so the residual-stream activations we capture next will encode it too. That is fine as
   long as it is **held fixed** across all prompts (it is) — but worth recording so M_y↔M_h pairing
   compares like with like.
5. **"other" semantics.** We treat 1 − Σweekday as a single "other" bin (paper convention). The
   retained set has other ≤ 0.10 by construction; the *shape* of the off-concept mass is discarded.
   Fine for simplex coverage; revisit if "other" structure matters later.
6. **Activation capture not yet done (by design).** Now that the prompt set + frame are validated,
   the next run should dump last-token residual activations for these exact 194 prompts (all layers)
   to start M_h — a cheap add-on to the same script.

## Reproduce

```bash
# prompts (pure python, local)
python code/methods/weekday_prompts/prompts.py
# coverage on cinaps (1 GPU; --mem REQUIRED or the job never schedules)
sbatch code/analyses/simplex_coverage/sbatch_full_base.sh           # base, the deliverable
sbatch code/analyses/simplex_coverage/sbatch_full.sh                # base + instruct comparison
```

Artifacts: `artifacts/weekday_simplex/{model}/simplex_coverage/{distributions.safetensors,
prompts_scored.json, coverage_metrics.json, figures/}`.

---

# Step 2 — behaviour subspace vs activation subspace (linear)

**Reframing (per user):** the session goal is to **explore/compare the subspaces**, not to fit
manifolds. We do not assume a low-D curved manifold captures the whole subspace (unlikely). So this
step compares the **behavioural subspace** (span of the Hellinger coords √p(x), ~6-D) with the
**activation subspace** (residual-stream h_ℓ(x)) **linearly**, per layer.

**Method.** Captured last-token residual activations at every layer for all 225 prompts on base
Llama-3.1-8B with the winning frame (one forward pass; `activations.safetensors`, shape
(225, 33, 4096)). On the 194 retained prompts, per layer: standardise → PCA(top-40); then (a) **CCA
canonical correlations** between the activation top-PCs and the behaviour coords = cosines of the
principal angles between the two subspaces; (b) **5-fold CV R²** of a linear decode
activation → behaviour coords; (c) activation effective dimension (participation ratio).
(`code/analyses/simplex_coverage/{capture_activations,subspace_compare}.py`, job 12423.)

**Result — the behavioural subspace is linearly embedded in the late residual stream.**

| layers | mean canonical corr | # canon > 0.9 (of 6) | linear-decode R²(CV) |
|---|---|---|---|
| 0–18 (early/mid) | 0.31 – 0.55 | 0 | **negative** (not linearly decodable) |
| ~19 (onset) | 0.55 → **0.82** | 2 | turns **positive** (0.36) |
| 23–28 | 0.90 – 0.94 | 3 – 5 | 0.57 – 0.76 |
| **29–32 (late)** | **0.95 – 0.965** | **6 / 6** | **0.81 – 0.86** |

- **Best layer L31:** all 6 canonical correlations > 0.9 (mean 0.965), linear-decode **R² = 0.86**.
  The behaviour and activation canonical variates are essentially the same coordinate (see
  `result/figures/best_layer_cca_scatter.png`): canonical pair 1 lays the days out as a continuum
  (Sat/Sun → Fri → Mon/Tue/Wed), pair 2 is a second axis.
- **Sharp onset at ~L19** (`subspace_alignment_by_layer.png`): the alignment is flat-low through the
  first ~18 layers then jumps, saturating near the output. The weekday subspace is a **mid-to-late**
  feature, not present at the input embedding.
- **The behaviour subspace is a ~6-D slice of a larger activation subspace.** Activation effective
  dimension at late layers is ~17–18 (participation ratio), vs the 6-D behavioural subspace — so only
  ~6 of the activation directions are behaviour-aligned; the other ~12 encode off-behaviour structure
  (prompt surface form, family, etc.). This is direct evidence that a 6-D manifold would **not**
  capture the activation variation — vindicating the decision to study subspaces, not manifolds.

**Step-2 flags.**
- The linear decode (R²≈0.86) is strong but not perfect — some behavioural variance is non-linear in
  activations and/or lost to the top-40-PC truncation. Worth a sensitivity sweep over PC count and a
  per-behaviour-dim R² breakdown.
- Activations include the fixed few-shot prefix and the colon stem; the ~12 non-behaviour activation
  dims plausibly track family/surface form. A controlled check: regress out family and re-measure.
- L0 R² is a degenerate fold artefact (huge negative) — read as "no signal," not a real number.

---

---

# Step 3 — mapping the activation subspace by perturbing anchors (causal)

**Framing (per user).** The 194 prompts are **anchors**: activation points we *know* decode to valid
weekday behaviour. To map the (assumed continuous) activation subspace directly, we **perturb the
last-token residual around each anchor at L29/30/31, patch it back through the model, and test
whether the output is still a valid weekday distribution** (Σweekday ≥ 0.90, other ≤ 0.10). This is
causal (intervene on activations → read behaviour), not decoding. No behaviour-space PCA.

**Design** (`perturb_map.py`, jobs 12424–12425):
- **Patch** = forward hook on decoder layer L−1 overwriting the last-token residual; the anchor's
  *own* prompt is the carrier (earlier-token context intact). **Sanity: zero-perturbation reproduces
  the anchor distribution** (Hellinger 0.002–0.015 across layers) → the patch is correct.
- **Directions:** random unit vectors either in the **anchor-spanned subspace** (PCA of the 194
  anchors, top-20) or in **raw R⁴⁰⁹⁶** (control). **Scale** in units of the median anchor-to-anchor
  distance (NN-dist ≈ 10–13), so radius "1" ≈ stepping toward a neighbouring belief state.
- Radial sampling (validity boundary) on all 194 anchors × 3 layers; a cumulative random walk on L31.

**Result — the valid weekday region is a continuous, low-dimensional, traversable subspace.**
(`result/figures/perturb_radial_all_layers.png`)

| radius (×NN) | subspace valid% | subspace Hellinger-moved | full valid% | full Hellinger-moved |
|---|---|---|---|---|
| 1.0 | 97–99% | 0.22–0.26 | 99% | 0.04 |
| 2.0 | 87–92% | 0.39–0.44 | 90–97% | 0.08–0.10 |
| 4.0 | 44–58% | 0.57–0.61 | 14–57% | 0.16–0.18 |
| 6.0 | 18–30% | 0.65–0.68 | 0–1% | — |

- **Behaviour-relevant variation lives in the anchor subspace (answers point 4 empirically, at all
  three layers).** In-subspace steps move behaviour **~5–6× more per unit distance** than random
  full-space steps (r=1: Hellinger 0.24 vs 0.04). Random full-space directions are mostly orthogonal
  to the read-out — they barely change behaviour and merely break the model once pushed far
  (validity → 0 by r≈6). **So walking the raw 4096-D is the wrong move; restrict to the anchor
  subspace.** (Reducing instead to the 6-D CCA subspace would be circular, so we used the broader
  ~20-D anchor span and let validity reveal the relevant directions.)
- **The region is sizable and connected.** In-subspace, ≥90% of perturbations stay valid out to
  ~2 NN-distances while reaching Hellinger ≈0.4 from the anchor — i.e. you reach substantially
  different *but valid* weekday distributions, and since valid radius (~2 NN) exceeds the anchor
  spacing (1 NN), neighbouring anchors' valid regions **overlap → one connected subspace**, not
  isolated islands. The cumulative random walk (L31, step 0.5 NN) confirms traversal: **90% of walks
  still valid after 10 steps, 56% after 30** (~15 NN cumulative).
- **Finite extent / graceful boundary.** Validity falls off beyond ~2–4 NN radially; in-subspace it
  degrades *gracefully* (still 17–30% valid at r=6, reaching Hellinger ≈0.7 — the far simplex),
  whereas full-space fails *catastrophically* (off-manifold → garbage). The three late layers behave
  almost identically (L29 slightly more robust at large radius).

**Step-3 flags / next refinements.**
- "Valid" (mass filter) ⊋ "natural": a perturbation can keep weekday mass high yet be a distribution
  the model would never emit. Next: also score naturalness (e.g. distance to the anchor cloud / a
  density model), and record *which simplex region* each valid perturbation lands in (are we
  uniformly covering the simplex, or only smearing near each anchor?).
- The ~20-D anchor subspace is broader than the ~6 behaviour-relevant dims — a per-direction sweep
  (which subspace axes move behaviour vs which keep it fixed) would isolate the true behaviour
  subspace dimensionality from inside this causal map.
- Boundaries are reported as aggregate rates; a per-anchor validity radius + its dependence on the
  anchor's simplex position (interior vs vertex) is the natural next cut.

---

---

# Step 4 — shape of the valid region: is the map onto?

**Question (per user).** Does walking the valid activation subspace **fill** the behavioural simplex
(map onto), or only smear locally around each anchor? And how **natural** are the reached points
(staying valid on the mass filter ≠ being a distribution the model would actually emit)?

**Method** (`perturb_map.py --mode shape`, L31, all 194 anchors × 24 in-subspace dirs × radii
{0.5,1,1.5,2,3}; `region_shape_analyze.py`). For each **valid** in-subspace perturbation, record the
simplex point reached and a **naturalness** score = distance to the nearest anchor in the anchor
subspace (in NN-dist units). Project anchors + perturbations onto a 2-D Hellinger plane (fit on
anchors — only for visualisation/coverage; behaviour itself is never PCA-reduced) and measure grid
coverage. 20 544 valid perturbations (88% of samples).

**Result — the map is largely ONTO: the activation subspace fills the behavioural simplex.**
(`result/figures/region_shape_overlay.png`, `region_shape_naturalness.png`,
`region_shape_metrics.json`)

- **7.8× coverage gain.** On a 40×40 Hellinger-plane grid, anchors occupy 159 cells; valid
  perturbations occupy **1234** (1075 of them **new**). **73% of valid perturbations land in simplex
  regions no anchor occupied** — walking densely fills the *interior between* anchors.
- **Same extent, denser fill (not blow-up).** Perturbation 2-D spread (std 0.36/0.36) ≈ anchor
  spread (0.36/0.35): perturbations fill the region the anchors outline rather than spilling beyond
  it. The valid activation region maps onto (essentially) the convex region the anchors span.
- **Reasonably natural.** Nearest-anchor distance is median **1.4** NN-dist (p90 2.7, max 3.0) — the
  filled points sit *between* real anchors, not in far-off activation territory; naturalness degrades
  smoothly toward the region's edges (right panel of the overlay figure).
- **All 7 day-regions filled** (perturbation arg-max: Mon 4292 … Fri 2999 … Thu 2153) — coverage is
  not confined to one corner.

**Reading.** Combined with Steps 2–3: the weekday concept occupies a **continuous, ~6-effective-dim,
behaviour-aligned subspace inside the late residual stream**; the discrete prompt-anchors are a
*sample* of it, and perturbing within the anchor span **interpolates the whole valid region**, which
**maps onto the behavioural simplex**. This is the behaviour↔activation **subspace map** the session
set out to find — established without assuming or fitting a manifold.

**Step-4 caveats.** (1) "Onto" is measured by 2-D grid coverage of a Hellinger projection; a
higher-fidelity onto-test would invert (target a simplex point, find an activation that yields it).
(2) Naturalness here is geometric (nearest-anchor); a model-based density (e.g. likelihood of the
activation under natural runs) would be stronger. (3) L31 only — L29/30 expected similar (Steps 2–3
were near-identical across the three).

---

---

# Step 5 — a concept-general mapping routine + generalisation to months (validated)

The mapping procedure is packaged as a **concept-parametrised routine**
(`code/methods/concept_core.py`, `code/analyses/simplex_coverage/{capture_concept,map_subspace}.py`):
given a token set Z + a prompt set, it captures activations, filters to the valid region, isolates
the behaviour-relevant low-D subspace at a layer, and densely samples it to map the valid activation
region — for ANY concept.

**Generalisation to months (Δ¹²) — validated 2026-06-07 (jobs 12441/12442):**
- 269 month prompts (9 families incl. seasons, days-in-month, ordinals) → **230/269 retained (86%)**,
  **all 12 months single-token and reachable, balanced**, **11/12 Hellinger-PCA dims**, interior 0.40
  — the month simplex is spanned as well as weekdays.
- Subspace map at L31: **carrier-sanity 0.009** (behaviour = function of activation holds for months
  too), valid-fraction-of-box 0.79, coverage-gain 6.2×. **The routine generalises.**

**Overnight batch (2026-06-07 night) — full tracking in [`run/OVERNIGHT_JOBS.md`](../run/OVERNIGHT_JOBS.md):**
six jobs (12437–12440, 12442, 12443) sweeping the weekday map across layers 16–31, dimensionality k,
and sampling margin; mid-layer causal perturbation maps; the full months pipeline; and a
highest-resolution + reseeded weekday map.

### Overnight results (reviewed 2026-06-08)

ON1/ON2/ON5/ON6 completed; ON3 (margin) and ON4 (mid-layer perturb) were truncated by SLURM time
limits (partial data, low priority — the trends are already settled by other jobs).

- **Continuity with depth — confirmed, and concept-independent**
  (`result/figures/overnight_depth.png`). For BOTH weekdays and months, map faithfulness
  (carrier-sanity) improves monotonically **L16≈0.40 → L20≈0.13 → L24≈0.04 → L31≈0.010**, and
  behaviour-relevant subspace alignment **jumps at ~L19** (mean canonical corr 0.64→0.85) and climbs
  to **~0.96–0.97 by L31**. The two concepts' curves nearly overlap — the geometry of *how the
  concept subspace forms with depth* is the same for Δ⁷ and Δ¹². The latest layers (L30–31) give the
  cleanest, most well-defined map, vindicating the "work at late layers" assumption.
- **Behaviour-relevant dimensionality ≈ |Z|−1** (`result/figures/overnight_ksweep.png`). Simplex
  coverage-gain saturates near **k≈4–6 for weekdays** (|Z|−1=6) and **k≈6–10 for months** (|Z|−1=11):
  the behaviour-relevant activation subspace scales with the simplex dimension, exactly as a faithful
  behaviour↔activation map should. (Above the knee, extra dims add a little invalid volume —
  valid-fraction drifts down — without improving coverage.)
- **The map is robust.** Reseeded L31 maps are essentially identical (carrier 0.011–0.012, valid
  0.80–0.81, cov-gain 7.62–7.67); the 120k-sample "complete" map matches the 20–40k runs → metrics
  are converged, not sample-starved.
- **The valid region is bounded by the anchors.** Pushing the sampling box beyond the anchor hull
  (margin 0.6→1.0→2.0) steadily lowers valid-fraction (weekdays 0.81→0.69; months 0.78→0.60→0.50)
  while coverage-gain holds — i.e. **there is little valid activation territory the prompt-anchors
  didn't already bound.** The anchors are a good basis for the complete map.
- **Months generalisation is comprehensive** (not just the L31 spot-check): months reproduces every
  weekday phenomenon — depth onset, k-scaling, onto-ness (~7× coverage gain), bounded region. The
  routine is concept-general.

---

# Step 6 — leave-a-region-out recovery (method validation)

**Test (user).** Delete *all* anchors in one behavioural region (e.g. every Wednesday-argmax prompt),
rebuild the behaviour-relevant subspace + sampling box from the REMAINING anchors only, then
walk/sample that map and ask: do we **recover** the deleted region (reach valid distributions back in
that corner)? If yes, the method generalises beyond its anchor sample; if no, the walking needs work.
A matched **random** hold-out (same count, scattered) is the control (random deletion should recover
trivially via interpolation; deleting a whole region is the hard extrapolation test).
(`recovery_test.py`, job 12446, L31, k=8, margin 1.5.)

**Result — the method recovers deleted regions** (`result/figures/recovery_{Mon,Wed,Sat}.png`):

| deleted region | # anchors removed | region cell-recovery | max P(target) recovered | proj. keeps-argmax | random-control recovery |
|---|---|---|---|---|---|
| Mon (dominant, largest) | 49 | **0.87** | 1.00 | 0.80 | 0.98 |
| Wed | 31 | **0.96** | 1.00 | 0.77 | 0.93 |
| Fri | 30 | **1.00** | 1.00 | 0.67 | 0.90 |
| Sat | 19 | **1.00** | 1.00 | 0.74 | 1.00 |

- **87–100% of the deleted region's behaviour cells are recovered** by walking the training-only map —
  reaching valid distributions with **P(deleted day) up to 1.0** (the vertex itself, not just the
  fringe), and **comparably to the random-deletion control**. Monday — the dominant default region,
  49/194 anchors — is the hardest and still recovers 87%.
- **67–80% of the deleted anchors keep their day when projected onto the training-only subspace** →
  the behaviour-relevant subspace built without the region still *spans the region's direction*. The
  concept subspace is genuinely continuous, not a set of per-anchor memorised points.
- **Conclusion: the map generalises.** Removing a whole region and reconstructing it from the rest
  works, so the anchors are a sufficient sample of a continuous subspace — the mapping method is
  validated. (Caveat: `projection_recon_keeps_argmax` is only meaningful for the region hold-out, not
  the random control, where the deleted anchors span many days.)

**Completeness (margin sweep, now complete via ON_finish):** valid-fraction and coverage fall
monotonically as the sampling box extends past the anchor hull
(margin 0.6/1.0/1.5/2.5/4.0 → valid 0.81/0.69/0.56/0.39/0.29; cov-gain 7.9→4.5) — i.e. there is
little valid activation territory beyond the anchors, so the anchor hull (+ small margin) already
bounds the complete valid region. Combined with Step 6, the anchors are both **sufficient** (regions
recover) and **near-complete** (little valid mass outside their hull).

---

# Step 7 — inverse map (sharp onto-test)

**Question.** Sharper than the coverage-based onto-test: pick TARGET simplex points and find an
in-subspace activation that produces each; measure achievable behavioural error (Hellinger to
target). (`inverse_map.py`, job 12450, L29 & L31, k=8.) Method: fit a linear forward map
(subspace coords → √behaviour) on the anchors, invert via pseudo-inverse to propose coords per
target, reconstruct + patch to read the true achieved distribution, then refine with a few random
local steps. Targets: 7 vertices, 21 two-token edges (50/50), the uniform centroid, 60 Dirichlet
interior + 40 sparse mixtures.

**Result** (L31; `result/figures/inverse_map_L31.png`):

| target type | median Hellinger | reached ≤0.10 | reached ≤0.20 |
|---|---|---|---|
| **vertex** (one-hot day) | **0.03** | **100%** | 100% |
| **edge** (50/50 two-day) | 0.15 | 24% | 86% |
| uniform (centroid) | 0.17 | 0% | 100% |
| interior (Dirichlet) | 0.20 | 0% | 50% |
| sparse mixture | 0.22 | 3% | 38% |

- **The map is onto the model's *natural* reachable set.** Any single-day distribution is hit
  essentially exactly (Hellinger 0.03, all 7), and 50/50 two-day edges are reachable (~86% within
  0.20) — so we can steer the activation to produce any vertex or pairwise-mixture belief.
- **Arbitrary interior mixtures have a residual ~0.20.** Not every point of the abstract simplex is
  a distribution the model naturally emits; the reachable set is a curved sub-region that densely
  covers the vertices/edges. (Part of the residual is the *linear*-inverse + light-refinement
  budget; a full optimiser would tighten interior points — a follow-up.)
- L29 ≈ L31 (vertices 0.06 vs 0.03) — late layers invert slightly better.

This complements Step 4/6: coverage-onto-ness (~7.5× cell gain) + region recovery (87–100%) say the
map fills the *reachable* simplex; the inverse map says that reachable set is **vertex/edge-dense
with a bounded interior residual**.

---

# Step 8 — sparse-anchor steering completeness ("prompt for K, steer all of Z")

**Question (steering goal).** For continuous steering across a gamut, can we anchor a SPARSE subset
and still reach valid activations for the UNPROMPTED regions? Keep only a few days' anchors, build
the subspace from those alone, and measure how many held-out days become reachable — sweeping
subspace dim **k** and method (CCA / PCA / diff). (`sparse_recovery.py`, jobs 12451/12453; L31,
margin 2.0; "recovered" = a valid sample reaches P(day) ≥ 0.5.)

**Result — sparse anchoring + enough dimensions recovers the whole set:**

| anchored (kept) | held-out (unprompted) | recovered @ k=4 | recovered @ **k≥8** | strength |
|---|---|---|---|---|
| {Mon, Thu} (2) | Tue,Wed,Fri,Sat,Sun (5) | 3/5 | **5/5** (all 3 methods) | P(day)≈**1.0** each, 100s of samples/day |
| {Mon,Wed,Fri,Sun} (4 alt.) | Tue,Thu,Sat (3) | 1/3 | **3/3** | — |
| {Mon,Tue,Wed,Thu} (4 contig.) | Fri,Sat,Sun (3, *extrapolation*) | 1/3 | **3/3** (cca/pca; diff @k≥12) | — |

- **From just 2 anchored days we reach P≈1.0 on all 5 unprompted days** (`result/figures/sparse_recovery_MonThu.png`)
  — and the *contiguous* case (held days all on one side → pure extrapolation, not interpolation
  between anchors) also recovers fully. So sparse prompting + the activation subspace covers the
  whole simplex, including its edges.
- **Dimensionality is the knob, exactly as hypothesised.** k=4 under-recovers (1–3 of N); **k≥8
  unlocks full recovery** across CCA/PCA/diff. The behaviour-relevant subspace must be given enough
  dimensions (~|Z|−1) to span the directions toward unprompted regions.
- **CCA is hard-capped at the behaviour dimension |Z|+1 (=8 for days).** Its k=12/20/40 requests
  collapse to 8 — fine for days, but **for a wide gamut needing >|Z|+1 effective directions, CCA
  cannot supply them; PCA/diff (pure activation variance) can** and still recover here. *This is the
  key thing to carry to colours: use a method that isn't capped at the behaviour dimension.*

**Takeaway for the end goal.** The map supports the steering use case: anchor a sparse set of a
concept and reach the entire continuum in activation space, provided the subspace has ~|Z|−1
dimensions and (for large |Z|) is built by a non-CCA method.

---

---

# Day 3 — colours, Gemma, and the steering demo (in progress)

Tracking: [`run/DAY3_JOBS.md`](../run/DAY3_JOBS.md). Pull everything offline:
`bash run/pull_results.sh`.

- **Months recovery was a false negative at k=8 — corrected to k=13.** Months (Δ¹²) needs the subspace
  dim ≈ |Z|−1 (≈11–13), not the weekday default 8. Sparse already showed this (9/10 recovered at k=13);
  recovery re-run at k=13 in job 12482.
- **Engineering fix (global):** the decode computed full-vocab logits at *every* position → a 24 GB OOM
  and much of the slowness. All patch-decode scripts now pass `logits_to_keep=1` (last position only) —
  no OOM, much faster.
- **Colour simplex (Llama):** of a 150-name candidate list, **80 colours are single-token**; curating to
  canonical colours gives **Δ⁴⁵** (45 single-token real colours), 127 anchors, 50% retained, ~10
  behaviour dims — usable, though messier than days/months (ambiguous object-words like Sky/Ocean/Jet
  are dropped). Full colour suite (map / sparse / recovery / inverse / perturb) run jobs C1–C4 + D2/D3.
- **Gemma tokenizer has *more* single-token colours — 94 vs Llama's 80** (strict superset, including
  canonical Magenta/Indigo/Maroon/Burgundy/Mauve/Khaki/Cobalt that are multi-token in Llama). So a
  richer colour gamut is available on Gemma (`colors_gemma.json`) — cross-model capture is a "when back"
  item.
- **Steering trajectory demo (the payoff, jobs D1/D2/D4):** `steer_trajectory.py` inverts the map at each
  waypoint of a target path (weekday cycle, month cycle, colour rainbow Red→…→Purple, Red→Blue sweep),
  patches the activation, and records the achieved behaviour. Figures `…/figures/steer_*.png` show the
  achieved trajectory tracking the target path through behaviour space — i.e. **continuously steering the
  model across the gamut by walking the activation subspace.** Results landing through the day.

---

# Step 9 — Steering head-to-head: linear vs spline-manifold vs subspace (+ generation check)

**The missing comparison vs the paper** (jobs 12513; `steer_compare.py`). On identical endpoint
pairs (all 42 weekday / 40 month ordered centroid pairs), identical 16 carrier prompts, K=20
waypoints, five strategies: **linear_full** (paper Eq. 1 — raw-centroid chord, full residual
replaced), **manifold** (paper Eq. 2 — periodic cubic spline through argmax-conditional centroids
in PCA-64, off-subspace residual preserved), **subspace_inv_full / _resid** (ours — M_y-spline
targets, linear-inverse to k-D CCA coords; full-replace vs residual-preserving), **linear_subspace**
(straight line in the k-D coords). Naturalness = paper's E_BC cumulative energy to M_y (tangent-plane
spline through behaviour centroids, A.4). Fluency = 24 greedy tokens from the steered state, ppl of
tokens 2..24 under the *unsteered* model.

| E_BC (mean±SE; ↓) | linear_full | manifold (paper) | subsp_inv_full | **subsp_inv_resid** | linear_subspace |
|---|---|---|---|---|---|
| weekdays L19 | 0.76±.04 | 0.29±.02 | 0.25±.01 | 0.50±.02 | 1.05±.04 |
| weekdays **L23** | 0.52±.03 | 0.45±.02 | 0.17±.00 | **0.09±.00** | 0.43±.03 |
| weekdays L27 | 0.55±.03 | 0.49±.02 | 0.15±.01 | **0.10±.00** | 0.46±.03 |
| weekdays L31 | 0.66±.03 | 0.70±.01 | 0.68±.01 | **0.46±.01** | 0.56±.03 |
| months L23 | 0.86±.03 | 0.88±.02 | 0.50±.01 | **0.33±.01** | 0.78±.02 |
| months L31 | 0.90±.02 | 0.92±.01 | 0.64±.01 | **0.51±.01** | 0.79±.02 |

- **Subspace-inverse steering is the most natural everywhere, by 2–5× at mid-late layers** —
  E_BC 0.09 (weekdays L23) vs 0.45 spline-manifold / 0.52 linear. Residual-preserving beats
  full-replace (except at L19). The paper's manifold-vs-linear gap replicates **at L19** (0.29 vs
  0.76) but **vanishes at L31** (0.70 vs 0.66) — with a dense anchor set, the spline through
  vertex centroids is no better than a chord at the last layers; the subspace chart is what matters.
- **Best steering layer is L23/L27, NOT L31** — all methods get *worse* at the output layer
  (E_BC 0.46–0.70). Cross-layer caveat to the session's L31-first habit, and exactly the
  user's instinct that "L31 is late to steer".
- **Teleportation replicates the paper**: max consecutive cyclic jump = 1.0 for manifold/subspace
  methods (strictly adjacent sweeps) vs 2.0 (weekdays) / 3.4 (months) for both linear variants;
  100% mass-filter validity for ALL methods (the valid region is forgiving — naturalness, not
  validity, is what separates strategies).
- **No model failure after steering.** Continuation ppl ≈1.55–1.76 for every method/layer; no
  repetition blow-ups (uniq ~0.6, max bigram repeat 3 = the few-shot QA format reasserting itself);
  samples read clean (e.g. steered mid-path Fri→Mon the model emits " Saturday\nA color: red\n…").
  Full-residual replacement at one position does not break the model at these radii.

# Step 10 — Gradient inverse: the 0.20 interior residual was the optimiser, not the model

Job 12516 (`inverse_opt.py`): optimise k-D subspace coords by Adam *through the frozen model*
(the paper's pullback machinery applied per-target), init = linear-inverse proposal.

| target type (wk L31) | linear+random (Step 7) | **gradient (Step 10)** |
|---|---|---|
| vertex | 0.030 / 100% ≤0.2 | 0.027 / 100% |
| edge | 0.152 / 86% | **0.066 / 100%** |
| uniform | 0.165 / 100% | **0.062 / 100%** |
| interior | 0.203 / 50% | **0.065 / 100%** |
| sparse mix | 0.218 / 38% | **0.074 / 100%** |

Weekdays L23 overall median 0.08; months L31 overall 0.07. **Step 7's "bounded interior residual
~0.20" is retracted as a model property** — with gradient pressure the in-subspace patch reaches
essentially any simplex point (median Hellinger ≈0.07). The map is onto the (near-)full simplex,
not merely the "natural reachable set"; what Step 7 measured was the linear-inverse + random-search
budget.

# Step 11 — The geometry of the map: a calibrated core inside a saturated shell

Per-sample dumps (job 12512, `map_geometry_dump.py`) + offline `analyze_geometry.py`;
figures `geometry_L{19,23,27,31}.png`, `geometry_isometry_layers.png`, `geometry_analysis.json`.
This answers "after PCA, everything in the box maps to valid behaviour — so where is the geometry?"

- **Two regimes, quantified** (weekdays L31; distances in units of median anchor NN-spacing):
  the anchor-fitted affine chart coords→√p has **R²=0.91 on anchors**, residual ~0.43 at 2–3 NN,
  0.88 at 4–6 NN, **1.25 at 6–10 NN** (≈ max possible 1.41). Meanwhile achieved entropy *drops*
  with distance (median 0.46 → 0.25 nats; near-vertex fraction → 56%). I.e. **the box stays
  "valid" far out, but the linear behaviour↔activation correspondence holds only on a core
  ≈≤2 NN around the natural states; beyond it the readout saturates into angular vertex sectors**
  (visible in the foliation figure: argmax organizes by *direction* from the core). Validity is
  blind to this — which is exactly why the all-green map figures looked structureless.
- **Steering lives in the core**: the inverse-map proposals for vertex-centroid targets sit
  0.5–1.6 NN from the anchor cloud — inside the calibrated chart. The box-margin sweeps (Step 5)
  sampled mostly ≥3 NN (8-D volume concentrates at the periphery), i.e. mostly the saturated shell.
- **The paper's ring is the skeleton of the core**: argmax-conditional centroids, projected onto
  their own principal plane, form the weekday cycle in **perfect cyclic order at L23/L27/L31
  (7/7)** — not yet at L19 (5/7) — and the month cycle **perfectly at both L23 and L31 (12/12)**.
  The 12-hue colour wheel is weaker and partial: 6/8 populated hues in cyclic order at L31, 4/8 at
  L23 — trained temporal cycles ring far more cleanly than the perceptual colour circle.
- **Core-merged chart-breakdown curve (G2)**: residual 0.23 at 0–1 NN, 0.33 at 1–2 NN, then an
  **entropy cliff at 2–3 NN** (median achieved entropy 0.86 → 0.10; near-vertex fraction 0.22 →
  0.70) and residual 0.88–1.25 beyond 4 NN. Graded interior belief states exist ONLY within ~2
  anchor-spacings of the natural set; outside, the readout saturates to vertices. The local
  isometry in the core also improves (r 0.31 → 0.45–0.53 at L31 with core samples).
- **Natural-state isometry strengthens with depth**: Euclid-vs-Hellinger correlation over anchor
  pairs r = 0.39 (L19) → 0.64 (L23) → 0.74 (L27) → **0.77 (L31)**; over the whole box it is only
  ~0.45–0.50 with a saturating distortion curve — approximate isometry is a property of the
  natural core, not of the valid region.
- **Jacobian drift confirms curvature at box scale**: local linear fits at random valid samples
  have principal angles of ~75–84° (median) to the global chart — the far-field map is a different
  (folded/saturated) regime, not a noisy version of the core chart.
- Core regime (0–2 NN) sampled directly in follow-up job G2 (12529, Gaussian-around-anchor dumps).

# Step 12 — Honest colour accounting: ceilings, reachable-set recovery, and the hue wheel

- **Natural ceilings** (`color_ceilings.py`, offline): **32/45 colours never exceed P=0.5 in any of
  the 255 colour prompts** (Charcoal/Rose/Plum/Peach ≈0.01). The sparse-recovery criterion
  (max P ≥ 0.5) was unreachable for 71% of the gamut from the start; the "flat ~10–19% in k" result
  mostly measured the model's output prior, not the map. **Among naturally-reachable held-outs
  (ceiling ≥0.5): keep-3 recovers 7/10, keep-8 recovers 3/5.** Persistent genuine misses: Navy,
  Violet (absorbed by neighbouring hue giants Blue/Purple — steering mass lands on the dominant
  synonym). Ceiling-normalized curves in `figures/color_ceilings.png`.
- **Colour region recovery was selectively reported**: alongside Blue 1.0 (weak test: 4 cells,
  5 anchors, blue-synonym anchors retained) and Green 0.74–0.83, **Red recovers only 0.35–0.70,
  non-monotonic in k** — now in the table. Family-exclusion deletions (whole blue/red families,
  no synonym leakage) running in job 12527.
- **hues12** — curated 12-colour wheel (Red…Pink, all single-token, hue-ordered; `hue_prompts/`,
  82 prompts): 47/82 retained; steering sweeps the wheel **in hue order** at L31 (argmax sequence
  Red→Orange→Yellow→Lime→Green→Cyan→Blue→…→Purple→Red) though mass-filter validity is low (0.40)
  because non-wheel colour tokens (Indigo, Turquoise, …) count as "other" under Δ¹² — a definition
  artifact to keep in mind. Ring test + sparse + recovery in job 12528.

# Day-4 overnight results (2026-06-12 morning) — conclusions closed out

All overnight jobs landed (A2a/A2b/N1/N1b/N2/N3; one index bug in the hues12 head-to-head fixed
and rerun as 12534). **Full synthesis → [`CONCLUSIONS.md`](CONCLUSIONS.md); publication figure set
F1–F6 → `result/figures/` (`make_final_figures.py`).** Headlines on top of Steps 9–12:

- **Significance (Wilcoxon, per-pair E_BC):** subspace-inverse (resid) beats BOTH linear and the
  spline manifold at **p ≤ 4.5×10⁻¹³ (weekdays) / 1.8×10⁻¹² (months)** at L23/L27/L31 (ratios
  1.5–5.9×). At L19 the spline/full-replacement variants win — the spline is the right tool at the
  onset layer, unnecessary later. *(F1, F6)*
- **Ablations complete:** weekdays **7/7 regions** (0.86–1.00; ≈ control), **{Mon,Tue,Wed} = 55%
  of anchors deleted → 1.00**, titration 25/50/75% flat; **months 12/12** (0.88–1.00, k=13);
  colour family-exclusion: hue giants recover (Blue/Red 0.998), shades don't (Navy 0.02 —
  ceiling/synonym-limited). *(F5)*
- **Unprompted-value steering (N1/N1b):** pure gradient from a sparse chart **stalls** (weekdays
  3/5; flat softmax basins) — the **hybrid explore-then-refine** inverse fixes it: weekdays
  keep{Mon,Thu} → **5/5 at P 0.81–1.0**; hue wheel keep{RGB} → 5/9; colours keep{RGB} → 15/42 incl.
  **8 shades steered ABOVE their natural ceiling** (Ruby 0.95 vs 0.12, Scarlet 0.86 vs 0.01) —
  steering reaches states outside the model's natural output repertoire. *(F4)*
- **Geometry robustness (N2/G2):** core-merged chart curve (resid 0.23 → 0.33 → entropy cliff at
  2–3 NN); seed replicate identical (iso r 0.458/0.486 vs 0.462/0.484, ring 7/7); months core
  local-iso 0.64, ring **12/12**; hue wheel ring 6/8 — temporal cycles ring cleaner than the
  colour wheel. *(F2, F3)*

**Remaining queue:** hue-wheel head-to-head (12534, folds into F1); then Gemma cross-model,
alphabet Δ²⁶ (sequential geometry), logit-lens mechanism of the saturated shell — future work.
