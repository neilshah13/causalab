# Day-3 batch (launched 2026-06-09 daytime; review ~7 h later)

Theme: **continuous steering** (the payoff) + finish colours. All Llama-3.1-8B, base, fewshot_neutral.
**To pull everything local + render figures: `bash agent_logs/<session>/run/pull_results.sh`** (from repo root).

> NB: jobs `pipe_weekdays_zh/ja`, `pipe_months_*` in `squeue` are **a labmate's** multilingual runs,
> not ours — they just share the 8 GPUs, so our jobs queue behind them and spread across the day.

## What to look at first (offline)

1. **Steering trajectory figures** — `artifacts/{weekday,months,colors}_simplex/llama31_8b/simplex_coverage/figures/steer_*.png`.
   Each shows a target path through behaviour space (black line) vs the **achieved** steered trajectory
   (rainbow points) over the anchors. This is the literal "steer the model across the gamut" demo.
2. **steer_*.json** — `fidelity_median_hellinger` (achieved vs target; lower=better), `smoothness_median_step`
   (consecutive achieved distance; small+uniform = smooth sweep), `frac_valid`, `achieved_argmax_seq`
   (does the arg-max sweep Mon→Tue→…→Sun / Red→Orange→…→Purple?).
3. **Colour coverage** — `colors_simplex/.../coverage_summary.json`: curated **45** single-token colours,
   127 anchors, 50% retained, ~10 behaviour dims (cleaner than the 80-colour set).

## Job table (12505–12508 = today; 12482–12487 = in-flight from before)

| Job | Name | What | Key outputs |
|---|---|---|---|
| **12505** | D1_steer_wk_mo | Steering weekdays (cycle) & months (cycle) across layers {19,23,27,31}/{23,27,31} × k | `…/steer_weekdays_L*k*.json/png`, `…/steer_months_L*k*.*` |
| **12506** | D2_steer_colors | Steering colours: **rainbow** Red→Orange→…→Purple (L27,31 × k16,24,32) + **Red→Blue** sweep | `colors_…/steer_colors_rainbow_*.*`, `…_RedBlue.*` |
| **12507** | D3_color_recov_hi | Colour **recovery** at adequate k {24,32,40} (Blue/Red/Green) + colour **sparse** (keep 3 & 8) | `colors_…/recovery_test_*_k*.json`, `sparse_recovery_*.json` |
| **12508** | D4_steer_hires | High-resolution steering trajectories (fine steps), all 3 concepts, L31 | `…/steer_*_hires.png` (the publication figures) |
| 12482 | R1_fixes | weekday sparse redo (OOM-fixed) + **months recovery at correct k=13** + months walk | `weekday_…/sparse_recovery_*.json`, `months_…/recovery_test_*_k13.json` |
| 12485 | C2_color_sparse | colour sparse-anchor steering (keep 4 & 6) | `colors_…/sparse_recovery_*.json` |
| 12486 | C3_color_recov_i | colour recovery (k=16 — likely under-dimensioned for Δ⁴⁵; see D3 for the right k) + inverse map | `colors_…/recovery_test_*.json`, `inverse_map_L31.json` |
| (done) | C1_color_map, C4_color_perturb | colour subspace map (layer/k sweep) + perturbation maps | `colors_…/map_subspace_metrics_*.json`, `perturb_*` |

## Key facts established just before this batch

- **Months generalisation holds** but recovery/sparse need **k ≈ |Z|−1** (months ~13, not the weekday 8) —
  the k=8 months-recovery run was a *false negative* (re-run at k=13 in R1).
- **OOM/slowness fixed globally:** decode now passes `logits_to_keep=1` (only last-position logits) →
  no more 24 GB OOM and ~much faster. All patch-decode scripts updated.
- **Gemma tokenizer: 94 single-token colours vs Llama's 80** (strict superset; incl. canonical
  Magenta/Indigo/Maroon/Burgundy/Mauve/Khaki/Cobalt that are multi-token in Llama). `colors_gemma.json`.

## How to read steering results

- **Good steering** = low fidelity Hellinger (achieved tracks target), small + uniform smoothness steps,
  high frac_valid, and `achieved_argmax_seq` that sweeps the intended order. Watch whether fidelity
  improves with **layer** (expect later better) and **k** (expect ≥|Z|−1 needed for colours).
- If colour steering fidelity is poor at k=16 but good at k=24–32 → confirms the dimensionality law on Δ⁴⁵.

## Next (when you're back)

1. Read the steering figures — does the model sweep the gamut smoothly? (the end-goal demo)
2. **Gemma cross-model**: capture Gemma activations + run the map (does the subspace story hold on a
   different architecture / its richer 94-colour simplex?). Needs base-vs-instruct + frame decision.
3. Gradient inverse optimiser to tighten the interior-mixture residual (better steering fidelity).
