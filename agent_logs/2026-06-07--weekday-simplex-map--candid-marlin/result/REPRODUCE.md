# REPRODUCE — weekday-simplex-map (behaviour↔activation subspace map)

How to regenerate every result in this session from the code in this repo. The session studies the
map between a concept's **behaviour simplex** (probability over single tokens — weekdays Δ⁷, months
Δ¹², colours Δ⁴⁵, a 12-hue wheel) and the **residual-stream activations** of Llama-3.1-8B, and shows
you can continuously steer the model across the gamut by walking that subspace.

> **What is and isn't in git.** The whole `agent_logs/` tree is git-ignored; this session is
> force-committed **text/code/data only**. Deliberately *excluded* (regenerable — see below):
> figures (`*.png/*.pdf/*.gif/*.svg`), slide decks (`*.pptx/*.zip`), geometry dumps (`*.npz`),
> captured activations (`*.safetensors`). Committed: all `code/`, the reports
> (`REPORT.md`, `OVERVIEW.md`, `CONCLUSIONS.md`), the SLURM launchers + logs (`run/`, `*.sh`, `*.out`),
> and every metric/results JSON under `artifacts/**/simplex_coverage/*.json`.

---

## 0. Prerequisites

**Compute.** Everything that touches the model runs on the **cinaps** SLURM cluster (see
`.claude/skills/running-on-cinaps/SKILL.md`). Non-negotiables learned the hard way:

- Work under `/workdir2/<login>/causalab` (never `$HOME`). Request GPUs with `--gpus=1`; **always set
  `--mem` and `--cpus-per-task`** (`--mem=96G --cpus-per-task=8`) — a job with no `--mem` defaults to
  the whole node's RAM and sits `PENDING` ~forever.
- Pre-download weights on the login node, run with `HF_HUB_OFFLINE=1` (compute nodes have no internet).
- Python env via `uv` installed into `/workdir2` (Python pinned 3.10); scripts run as `uv run python …`.

**Model.** `llama31_8b` (base) unless noted; prompt frame `fewshot_neutral`. The model→HF-id map lives
in each entry point's `MODEL_HF` dict — add a key there to run a new model (e.g. Gemma).

**Local (laptop).** To pull results and rebuild figures/PDFs you only need `python3` with
`pip install --user pymupdf matplotlib reportlab pillow`.

**Concepts** are registered in `code/methods/concept_core.py` → `CONCEPTS` /
`get_concept(name)`: `weekdays`, `months`, `colors` (= curated canonical colours ∩ single-token set),
`hues12` (curated 12-hue wheel). Each concept's carrier stem differs (weekdays `"A day of the week:"`,
months `"A month of the year:"`, colours/hues `"A color:"`).

---

## 1. Build the prompt sets (local, pure Python — no GPU)

```bash
S=agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
python3 $S/code/methods/weekday_prompts/prompts.py   # -> weekday_prompts/prompts.json (225)
python3 $S/code/methods/month_prompts/prompts.py     # -> month_prompts/prompts.json   (269)
python3 $S/code/methods/color_prompts/build.py       # -> color_prompts/prompts.json + colors.json
python3 $S/code/methods/hue_prompts/prompts.py       # -> hue_prompts/prompts.json     (82)
```

The colour simplex Z is defined by the tokenizer: `color_prompts/color_tokens.py` discovers
single-token colour names (Llama vs Gemma differ — `colors.json` vs `colors_gemma.json`).

## 2. Run the pipeline on cinaps

`rsync` the session to `/workdir2/<login>/causalab/agent_logs/<session>/`, then submit the numbered
launchers in `code/analyses/simplex_coverage/`. Each is a self-contained `sbatch` script that writes
to `artifacts/<concept>_simplex/<model>/simplex_coverage/`. The canonical order (launcher → what it
produces → REPORT step):

| Launcher(s) | Entry point(s) | Produces | Step |
|---|---|---|---|
| `sbatch_probe.sh` | `run_coverage.py --mode probe` | frame/format probes (`format_probe*.json`) | 1 (frame) |
| `sbatch_full_base.sh`, `sbatch_full.sh` | `run_coverage.py --mode full` | `coverage_metrics.json`, `prompts_scored.json`, `distributions.safetensors` | 1 |
| `sbatch_capture.sh` | `capture_concept.py` | `activations.safetensors` (all layers) | 2 |
| `sbatch_mapsub.sh` | `subspace_compare.py`, `map_subspace.py` | `subspace_compare.json`, `map_subspace_metrics*.json` | 2–5 |
| `sbatch_perturb.sh`, `sbatch_perturb_all.sh` | `perturb_map.py`, `region_shape_analyze.py` | `perturb_*`, `region_shape_metrics.json` | 3–4 |
| `sbatch_ON_{layersweep,ksweep,marginsweep,months,perturb}.sh` | `map_subspace.py` (swept) | depth/k/margin sweeps; months pipeline | 5 |
| `sbatch_recovery.sh`, `N1_months_recov.sh`, `A2a/A2b_ablation_*.sh` | `recovery_test.py` | `recovery_test_*.json` (leave-a-region-out) | 6 |
| `sbatch_inverse.sh`, `I1_inverse_opt.sh` | `inverse_map.py`, `inverse_opt.py` | `inverse_map_*`, `inverse_opt_*` | 7, 10 |
| `sbatch_sparse.sh`, `N4_wk_sparse.sh` | `sparse_recovery.py` | `sparse_recovery_*.json` | 8 |
| `D1–D6_*.sh` | `steer_trajectory.py`, `steer_gen_demo.py` | `steer_<concept>_*` trajectories | Day-3 |
| `S1_steer_compare.sh`, `N3_steer_compare_fin.sh`, `N3b_steercmp_hues.sh` | `steer_compare.py` | `steer_compare_*.json` (head-to-head vs paper) | 9 |
| `G1_geometry_dump.sh`, `G2_geometry_core.sh` | `map_geometry_dump.py`, `analyze_geometry.py` | `geometry_*.npz`, `geometry_analysis.json` | 11 |
| `C0–C4_*.sh` | colour capture/map/sparse/recovery/perturb | `colors_simplex/…` | 12 |
| `color_ceilings.py` (offline) | `color_ceilings.py` | `color_ceilings.json` (natural ceilings) | 12 |
| `H1_hues12.sh`, `H2_hues12_finish.sh` | full hue-wheel pipeline | `hues12_simplex/…` | 12 |

**Worked example — weekdays end to end** (what the headline numbers come from):

```bash
SDIR=/workdir2/$USER/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
sbatch $SDIR/code/analyses/simplex_coverage/sbatch_full_base.sh   # Step 1 coverage
sbatch $SDIR/code/analyses/simplex_coverage/sbatch_capture.sh     # Step 2 activations
sbatch $SDIR/code/analyses/simplex_coverage/sbatch_mapsub.sh      # Steps 2–5 subspace + map
sbatch $SDIR/code/analyses/simplex_coverage/sbatch_recovery.sh    # Step 6 region recovery
sbatch $SDIR/code/analyses/simplex_coverage/S1_steer_compare.sh   # Step 9 head-to-head vs paper
sbatch $SDIR/code/analyses/simplex_coverage/I1_inverse_opt.sh     # Step 10 gradient inverse
sbatch $SDIR/code/analyses/simplex_coverage/G1_geometry_dump.sh   # Step 11 geometry
```

The entry points are plain `argparse` (not Hydra); every launcher shows the exact flags. Generalising
to a new concept = build its prompts, then pass `--concept <name> --prompts <…/prompts.json>
--experiment-root <…/artifacts/<name>_simplex/<model>/simplex_coverage>` to the same scripts (this is
how months/colours/hues12 reuse the weekday code unchanged).

## 3. Pull results back + render figures (local)

```bash
bash agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/pull_results.sh
```

`pull_results.sh` rsyncs `artifacts/`, `run/` and the colour JSON from cinaps, then renders every
figure PDF to PNG (via `pymupdf`) for offline viewing. Edit the `REMOTE`/`SSHCFG` vars at the top for
your login.

## 4. Regenerate figures and the report PDFs

All figures are produced from the JSON/npz artifacts — no model needed:

```bash
S=agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
# publication figures F1–F6 (head-to-head, geometry, rings, unprompted, ablation, where-to-steer)
python3 $S/code/analyses/simplex_coverage/make_final_figures.py
# the two narrative PDFs
python3 $S/run/build_report_pdf.py                                  # -> result/weekday-simplex-map-RESULTS.pdf (Steps 1–8 + Day-3)
python3 $S/code/analyses/simplex_coverage/build_conclusions_pdf.py  # -> result/CONCLUSIONS.pdf
python3 $S/code/analyses/simplex_coverage/make_deck.py              # -> result/slides/*.pptx
```

Per-analysis figures are emitted next to their metrics under
`artifacts/<concept>_simplex/<model>/simplex_coverage/figures/`.

## 5. Output layout

```
artifacts/<concept>_simplex/<model>/simplex_coverage/
├── activations.safetensors        # captured residual stream (git-ignored; regenerate via capture)
├── coverage_metrics.json          # Step 1 coverage
├── subspace_compare.json          # Step 2 CCA / linear-decode by layer
├── map_subspace_metrics*.json     # Steps 3–5 map (k/layer/margin sweeps)
├── perturb_*.json, region_shape_metrics.json   # Steps 3–4
├── recovery_test_*.json           # Step 6
├── inverse_map_*.json, inverse_opt_*.json       # Steps 7, 10
├── sparse_recovery_*.json         # Step 8
├── steer_*.json, steer_compare_*.json           # Day-3, Step 9
├── geometry_analysis.json, geometry_*.npz       # Step 11 (npz git-ignored)
├── color_ceilings.json            # Step 12 (colours)
└── figures/                       # all PNG/PDF (git-ignored; regenerate as above)
```

## 6. Read the results

`result/CONCLUSIONS.md` (the claims) · `result/REPORT.md` (all 12 steps, every number) ·
`result/OVERVIEW.md` (headline table across Δ⁷/Δ¹²/Δ⁴⁵) · `run/DAY3_JOBS.md`,
`run/OVERNIGHT_JOBS*.md` (job tables + how to read).
