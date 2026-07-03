# Issues & findings — weekday-simplex-map

## 1. Prompt FRAME dominates weekday-mass and spread (resolved)

Eliciting high-weekday-mass *and* spread distributions was entirely a function of the prompt frame,
not the prompt content. Trajectory across probes/runs (jobs 12418–12422):

- `Q:\nA:` or bare stem, base → **0%** pass: argmax is the right day (98%) but 30–50% mass bleeds to
  prose lead-ins (` I`, ` The`, ` a`, ` called`).
- Copular "…is"/"…on" stems leak on base (prose continuation); **answer-first colon "…:"** stems fix
  it (relational/ordinal families E/G/H went ~0% → ~100% retained).
- **Few-shot must be neutral.** A few-shot of only-day exemplars, or an instruction "reply with a
  day", **leaks the answer space** (inflates mass artificially). Fix: neutral few-shot demonstrating
  terse *answer-first* completions on NON-weekday tasks (color/fruit/letter) — same mass, no leak.
- **Few-shot must be terse.** Adding continuations ("…, for example") raised off-concept 'other'
  mass ~0.05 → ~0.19 and pushed the spread families (B/C/I) over the 10% line (run 12421: 78
  retained, 0.19 interior). Reverting to terse colon few-shot (run 12422): **194 retained, 0.39
  interior**.

## 2. Instruct + "one word" collapses the simplex to vertices (decided against)

`llama31_8b_instruct` + "Answer in one word" gives 98% pass but **interior fraction 0.26** — it
piles points on the 7 vertices (e.g. "day after Monday" → Tue=1.00, no neighbour spread). The base
model preserves graded belief (interior 0.39, neighbour spread intact). Chose **base** for spanning
the interior + continuity with the manifold-steering activation work. (User steer: don't force
one-word answers; only need the day to be the first token — base + neutral few-shot honours this.)

## 3. cinaps: always set `--mem` (reinforced by user)

No `--mem` → job defaults to a full node's RAM (~773 GB) → sits PENDING ~1 year out, never runs. All
sbatch scripts here use `--mem=96G --cpus-per-task=8 --gpus=1`. (Already in the cinaps-access memory;
user re-flagged it.)

## 4. Instruct weights were not pre-cached

Only `Llama-3.1-8B` was in the HF cache; `Llama-3.1-8B-Instruct` had to be pre-downloaded on the
login node (internet) before the offline probe. Stored HF token (huggingface_hub) made the gated
download work without setting HF_TOKEN.

## 5. Local PDF rendering needs a workaround

`Read` on PDFs needs `pdftoppm` (not installed locally) and `uv` is not on the non-interactive PATH.
Rendered figures to PNG via `python3 -m pip install --user pymupdf` (fitz) for inspection.

## 6. Step 3 — activation-subspace perturbation map (done, L29/30/31)

Patching sanity passed (zero-perturbation Hellinger 0.002–0.015 → hook targets the right
layer/position). Random full-space perturbations move behaviour ~5–6× LESS than anchor-subspace
perturbations and break the model when pushed (validity→0 by ~6 NN-dist) → confirms the behaviour
subspace must be explored within the anchor span, not raw R^4096 (point 4). Valid region is
continuous (≥90% valid to ~2 NN-dist, overlapping anchor neighbourhoods; random walk 56% valid at 30
steps). CAVEAT to address next: "valid" (mass filter) ⊋ "natural" — need a naturalness score and to
record which simplex region each valid perturbation reaches (is the map onto?).

## 7. Steps 4-5 + overnight batch (2026-06-07 night)

- Step 4 (shape): the activation subspace maps ONTO the simplex (7.8× coverage gain, 73% of valid
  perturbations in new cells), same extent as anchors, naturalness ~1.4 NN.
- Step 5 (routine + generalisation): concept-general routine built (`concept_core.py`,
  `capture_concept.py`, `map_subspace.py --concept`); **months (Δ¹²) validated** — 230/269 retained,
  11/12 PCA dims, all months reachable, L31 carrier-sanity 0.009.
- 6 overnight jobs launched (12437-12440, 12442, 12443) — full manifest in `run/OVERNIGHT_JOBS.md`.
- CAVEAT: `perturb_map.py` (radial/walk/shape) is still weekday-hardcoded — NOT generalised to
  `--concept` yet (map_subspace + capture were). Months only gets the subspace-map, not the causal
  perturbation maps, until perturb_map is generalised (tomorrow).

## 8. Recovery validation (2026-06-08) — method validated

Leave-a-region-out test (job 12446): delete all anchors of a behavioural region, rebuild map from
rest, recover? Deleted regions recover **87% (Mon, hardest) → 100% (Fri/Sat)** of cells, reaching
P(target)=1.0, ~ random-deletion control. Projection of deleted anchors onto the training-only
subspace keeps the day 67–80% of the time → subspace spans the deleted direction. Margin sweep
(finished via 12447) confirms the valid region is bounded by the anchor hull (valid 0.81→0.29 over
margin 0.6→4.0). Conclusion: anchors are a sufficient + near-complete sample of a continuous subspace.

## 9. Steps 7-8 + night-2 batch (2026-06-08)

- Step 7 inverse map: vertices hit to Hellinger 0.03 (all reachable), edges ~0.15, interior ~0.20
  (map onto the model's *natural* reachable set).
- Step 8 sparse-anchor steering: keep {Mon,Thu}→recover 5/5 unprompted days at k≥8 (P≈1.0);
  {Mon,Wed,Fri,Sun}→3/3; contiguous {Mon,Tue,Wed,Thu}→3/3 (extrapolation). k≥8 is the threshold;
  **CCA caps at |Z|+1** (use PCA/diff for larger gamuts like colours).
- `perturb_map.py` GENERALISED to `--concept` (was the last weekday-only script).
- Night-2 queue (12455-12459): months full treatment (recovery/inverse/sparse/perturb), weekday
  sparse deep-dive, colour prep. Manifest: `run/OVERNIGHT_JOBS_2.md`.
- Night-1 lesson re-applied: `sparse_recovery` is slow → leaner samples + generous `--time`.

## 10. Day-4 batch (2026-06-11) — runtime + infra lessons

- **recovery_test costs ~60–90 min per run, not minutes** (2×25–30k patched decodes + model load;
  slower on node15/node20-class GPUs). A1 (23 runs, 6h) and H1 (4h) hit TIME LIMIT; resubmitted as
  A2a/A2b/H2 with leaner samples (12–15k) and 10–11h limits. Budget ~20 min per 30k decodes on
  node22, ~2× on slower nodes.
- **`sacct` is disabled on cinaps** ("Slurm accounting storage is disabled") — job-completion
  monitoring must poll `squeue` + grep the `.out` for DONE/CANCELLED markers. An empty transient
  `squeue` response looks identical to "job left the queue" — require the .out marker before
  declaring a job done.
- **macOS Accelerate emits spurious `divide by zero/overflow in matmul` RuntimeWarnings** on
  float32 lstsq/matmul in the offline analysis — harmless (float64 gives identical numbers);
  silenced by casting to float64 on load.
- **steer_trajectory/recovery 'valid' is concept-definition-sensitive**: for hues12 (Δ¹² wheel),
  mass on NON-wheel colour tokens (Indigo, Turquoise…) counts as "other", so wheel-steering shows
  valid≈0.40 even when the argmax sweeps the wheel perfectly in order. Compare like with like.

## 11. Step-7 interpretation retracted (2026-06-11)

Step 7 claimed a bounded interior residual (~0.20) = "the model's natural reachable set".
`inverse_opt.py` (gradient through the frozen model) reaches the same targets at median ≈0.065–0.074
(100% ≤0.2) — the residual was the linear-inverse + random-refinement budget. REPORT Step 10.

## Open / deferred

- Inverse-map / sharp onto-test (optimise activation → target simplex point). [partly done Step 7; gradient version still TODO]
- Generalise `perturb_map.py` to `--concept`.
- Stronger naturalness (activation likelihood, not nearest-anchor).
- 3rd concept (alphabet Δ²⁶) to stress-test larger |Z|.
- Housekeeping: weekday prompts ≥200 retained; prune weak `D_letter` paraphrase.
