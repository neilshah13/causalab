# Overnight jobs — NIGHT 2 (launched 2026-06-08 evening; review 2026-06-09)

Balanced batch: mostly **months full-treatment** + **weekday sparse deep-dive**, with **colour =
prep only** (per instruction not to overindex on colour). All on cinaps, base llama31_8b,
frame fewshot_neutral, `--mem=96G --cpus-per-task=8 --gpus=1`. Artifacts on cinaps under
`…/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/artifacts/{weekday,months,colors}_simplex/llama31_8b/simplex_coverage/`.
`agent_logs/` is gitignored → rsync back in the morning.

## Morning routine

```bash
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
ssh -F ~/cinaps_ssh_config cinaps 'squeue -u johan.boscher; sacct --format=JobID,JobName%18,State,Elapsed -S 2026-06-08'
rsync -az -e "ssh -F ~/cinaps_ssh_config" cinaps:/workdir2/johan.boscher/causalab/agent_logs/$SESSION/artifacts agent_logs/$SESSION/
rsync -az -e "ssh -F ~/cinaps_ssh_config" cinaps:/workdir2/johan.boscher/causalab/agent_logs/$SESSION/run agent_logs/$SESSION/
grep -L "N[0-9] DONE" agent_logs/$SESSION/run/N*_*.out   # any file without a DONE = failed/truncated
```

## Job table

| Job | Name | Script | What it does | Key outputs (under the concept's `simplex_coverage/`) | Question |
|---|---|---|---|---|---|
| **12455** | N1_months_recov | `N1_months_recov.sh` | months **recovery** (hold out Jul/Jan/Apr) + **inverse map** (L31,L29) | `months_…/recovery_test_{Jul,Jan,Apr}.json`, `inverse_map_L{31,29}.json`, `figures/` | Do region-recovery + onto-ness hold on Δ¹²? |
| **12456** | N2_months_sparse | `N2_months_sparse.sh` | months **sparse-anchor steering**: keep {Jan,Jul} & {Jan,Apr,Jul,Oct}, k∈{4,8,13,20,40}×{cca,pca,diff} | `months_…/sparse_recovery_{JanJul,JanAprJulOct}.json`, `figures/sparse_recovery_*.png` | Does "prompt for K, steer all" hold on Δ¹²? Does the CCA cap (=13) bite? |
| **12457** | N3_months_perturb | `N3_months_perturb.sh` | months **causal perturbation** maps (radial+shape+walk, L31) — `perturb_map` now `--concept` | `months_…/perturb_{radial,walk}_L31.json`, `region_shape_L31.npz` | Continuous/traversable valid region on Δ¹²? |
| **12458** | N4_wk_sparse | `N4_wk_sparse.sh` | weekday sparse deep-dive: keep **{Mon}** (1 day!), **{Sat,Sun}** (weekend-only), **{Mon,Tue,Wed,Thu}**; finer k∈{3,5,6,7,8,12} | `weekday_…/sparse_recovery_{Mon,SatSun,MonTueWedThu}.json` | Pin the k-threshold; can 1 anchored day reach the other 6? does weekend→weekday work? |
| **12459** | N5_color_prep | `N5_color_prep.sh` | **COLOUR prep**: `color_tokens.py`→`colors.json` (single-token colours) → `build.py`→colour prompts → `capture_concept` → validation `map_subspace` (L31) | `code/methods/color_prompts/{colors.json,prompts.json}`, `colors_…/{coverage_summary.json,activations.safetensors,map_subspace_metrics_validate.json}` | Which colours are single-token? Does the colour simplex behave like days/months? |

## How to read

- **recovery_test_*.json**: `region_holdout.cell_recovery_rate` (want high), `…max_P_target_recovered`,
  `projection_recon_keeps_argmax_frac`; vs `random_control`. (Step-6 keys.)
- **sparse_recovery_*.json**: per (method,k) `frac_held_recovered` + per-day `max_P`. Watch the
  **k-threshold** and whether **pca/diff exceed the CCA |Z|+1 cap** (=8 days / 13 months).
- **inverse_map_L*.json**: achievable Hellinger by target type (vertex/edge/interior). (Step-7 keys.)
- **N5 coverage_summary.json**: retained, PCA dims, all-tokens-reachable for the colour simplex.

## Code added tonight (all py_compiled)

- `methods/concept_core.py` → `get_concept("colors")` loads the discovered set from `colors.json`.
- `methods/color_prompts/{color_tokens.py, build.py}` → colour vocabulary discovery + prompts.
- `analyses/simplex_coverage/perturb_map.py` → **generalised to `--concept`** (was weekday-only);
  months/colours now get radial/walk/shape maps too.
- (already concept-general: capture_concept, map_subspace, recovery_test, sparse_recovery, inverse_map.)

## Status of Step 8 (done before queuing) — context

Weekday sparse recovery: {Mon,Thu}→5/5, {Mon,Wed,Fri,Sun}→3/3, {Mon,Tue,Wed,Thu}(contiguous)→3/3,
all at **k≥8**; CCA capped at |Z|+1=8. Written up in `result/REPORT.md` §Step 8.

## Caveats / watch-outs

- `sparse_recovery` is SLOW (~1–2 min per 20k-sample config; many configs). Night-1 truncated two
  jobs on time limits → tonight uses leaner samples (10–12k) + generous `--time` (3–5 h). If N2/N4
  still truncate, the per-keep JSONs that DID finish are valid (each keep writes its own file).
- N5 colour simplex size is unknown until `color_tokens.py` runs; all scripts handle arbitrary |Z|.
- Colour analysis suite (recovery/sparse/inverse/perturb on colours) is deliberately NOT queued —
  do it tomorrow once we see the colour coverage, using PCA/diff (not CCA) for the larger gamut.
