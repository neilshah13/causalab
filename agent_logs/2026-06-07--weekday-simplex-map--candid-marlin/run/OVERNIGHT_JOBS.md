# Overnight jobs — launched 2026-06-07 night (review 2026-06-08 morning)

## STATUS (reviewed 2026-06-08) ✅ pulled + analysed
- **ON1 (12437) layersweep — DONE.** ON2 (12438) ksweep — DONE. ON5 (12442) months — DONE (full pipeline). ON6 (12443) complete+seeds — DONE.
- **ON3 (12439) marginsweep — PARTIAL** (TIME LIMIT) → **FINISHED via ON_finish (12447)**: full margin sweep 0.6→4.0 now present (valid drops 0.81→0.29; valid region bounded by anchor hull).
- **ON4 (12440) perturb — PARTIAL** (TIME LIMIT) → **ON_finish (12447)** added radial L27 + walks L25/L29 + shape L27.
- **NEW: recovery_test (12446) — DONE.** Leave-a-region-out validation (Mon/Wed/Fri/Sat hold-outs + random control). Result: deleted regions recover 87–100% → method validated. See `result/REPORT.md` §Step 6, `recovery_test_*.json`, `figures/recovery_*.png`.
- Findings written into `result/REPORT.md` §"Overnight results"; figures `result/figures/overnight_{depth,ksweep}.png`.
- Fix for any re-run: each `map_subspace`/`perturb_map` call reloads the 8B (~1-2 min) and the sweeps had too many iterations for the `--time` budget → split sweeps into more jobs or raise `--time`.

---


All on cinaps, base **llama31_8b**, frame **fewshot_neutral**, `--mem=96G --cpus-per-task=8 --gpus=1`.
Artifacts live on cinaps under
`/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/artifacts/`
and the slurm logs under `…/run/<jobname>_<jobid>.out`. **`agent_logs/` is gitignored — rsync back in the morning.**

## Morning routine (do this first)

```bash
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
# 1) check all finished
ssh -F ~/cinaps_ssh_config cinaps 'squeue -u johan.boscher; sacct --format=JobID,JobName%18,State,Elapsed -S 2026-06-07'
# 2) pull everything back
rsync -az -e "ssh -F ~/cinaps_ssh_config" \
  cinaps:/workdir2/johan.boscher/causalab/agent_logs/$SESSION/artifacts agent_logs/$SESSION/
rsync -az -e "ssh -F ~/cinaps_ssh_config" \
  cinaps:/workdir2/johan.boscher/causalab/agent_logs/$SESSION/run agent_logs/$SESSION/
# 3) check each job's log ended with its DONE marker (ON1..ON6 DONE / MONTHS_VAL DONE)
grep -L "DONE" agent_logs/$SESSION/run/ON*_*.out   # any file WITHOUT a DONE = failed/incomplete
```

Render any PDF figure to PNG to view: `python3 -c "import fitz; fitz.open('X.pdf')[0].get_pixmap(matrix=fitz.Matrix(2,2)).save('X.png')"`.

## Job table

| Job ID | Name | Script | What it does | Key outputs (under `artifacts/{concept}_simplex/llama31_8b/simplex_coverage/`) | Question it answers |
|---|---|---|---|---|---|
| **12437** | ON1_layersweep | `sbatch_ON_layersweep.sh` | weekday subspace map at layers **16,18,20,22,24,26,28,30,31** (k=8, 40k samples) | `map_subspace_metrics_layersweep.json`, `figures/map_subspace_layersweep_L*.png` | How do carrier-faithfulness + valid-region completeness evolve with depth? (continuity-with-depth) |
| **12438** | ON2_ksweep | `sbatch_ON_ksweep.sh` | weekday map at L25 & L31 for **k ∈ {2,3,4,5,6,8,10,12}** (20k) | `map_subspace_metrics_k{K}.json` (8 files) | Where does mapping completeness saturate in k? (expect ~6 = |Z|−1) |
| **12439** | ON3_marginsweep | `sbatch_ON_marginsweep.sh` | weekday map at L31, **margin ∈ {0.6,1.0,1.5,2.5,4.0}** (30k) | `map_subspace_metrics_margin{M}.json` (5 files) | Do valid activation regions exist BEYOND the anchor hull? (completeness beyond prompts) |
| **12440** | ON4_perturb | `sbatch_ON_perturb.sh` | causal perturbation maps at **L25,L27** (radial+shape) and walks at **L25,L29** | `perturb_radial_L25.json`,`perturb_radial_L27.json`,`perturb_walk_L25.json`,`perturb_walk_L29.json`,`perturb_shape_L25.json`,`perturb_shape_L27.json`, `region_shape_L*.npz` | Mid-layer validity boundary + shape (complements L29-31) |
| **12442** | ON5_months | `sbatch_ON_months.sh` | **MONTHS generalisation**: capture + layer sweep (16-31) + k-sweep (L31, k 3-12) + margin sweep (L31) | `months_simplex/.../coverage_summary.json`, `activations.safetensors`, `map_subspace_metrics_{layersweep,k*,margin*}.json`, `figures/` | Does the routine generalise to a 12-token simplex (Δ¹²)? |
| **12443** | ON6_complete | `sbatch_ON_complete.sh` | weekday highest-res maps L29&L31 (120k, k6, margin1.0) + reseeds (L31 seed1,seed2) | `map_subspace_metrics_complete.json`, `_seed1.json`, `_seed2.json` | The "most complete" single-layer maps + map stability across seeds |

## How to read each metric (per `map_subspace_metrics*.json`, per layer)

- `carrier_sanity_median_hellinger` — patching an anchor's own activation into the fixed carrier vs its true behaviour. **≈0 ⇒ behaviour is a function of the activation alone (map well-defined).** (validated: L31≈0.01, L19≈0.13.)
- `cca_canonical_corr` — alignment of the behaviour-relevant subspace per dim (cosines of principal angles).
- `valid_fraction_of_box` — fraction of the sampled k-D subspace box that decodes valid (size of V).
- `coverage_gain_ratio`, `behaviour_cells_new` — how much MORE of the simplex the map fills vs the anchors (onto-ness; ~7.5× weekdays, ~6× months so far).
- `argmax_day_counts_valid` — are all concept tokens reached?

For ON5 months also read `coverage_summary.json` (already validated: 230/269 retained, **11/12 PCA dims**, all months reachable).

## Already-completed today (context, not overnight)

- 12418/12419 frame probes; 12420/12421/12422 weekday coverage (winner: base+fewshot_neutral, 194/225, see `result/REPORT.md` Step 1).
- 12423 weekday capture + Step-2 subspace CCA (L31 R²0.86, all-6 corr>0.9).
- 12424/12425 Step-3 perturbation maps (L29/30/31 + walk). 12434 Step-4 shape. 12435/12436 map_subspace validation + L19/23/27/31 sweep. 12441 months validation.
- Full narrative + figures: `result/REPORT.md` (Steps 1-4), `issues.md`.

## Open follow-ups for tomorrow (interactive, NOT queued)

1. **Inverse map / sharp onto-test** — optimise an in-subspace activation to hit target simplex points (incl. interior gaps); measure achievable Hellinger. (New script — write interactively.)
2. **Generalise `perturb_map.py` to `--concept`** (only map_subspace/capture were generalised) so months gets the causal radial/walk/shape maps too.
3. Stronger **naturalness** (activation likelihood under natural prompts) vs the current geometric (nearest-anchor) score.
4. A **3rd concept** (e.g. alphabet A-Z, Δ²⁶) to stress-test generality at larger |Z|.
5. Housekeeping: top weekday prompts up to ≥200 retained; prune the weak `D_letter` "beginning with {L}" paraphrase.
