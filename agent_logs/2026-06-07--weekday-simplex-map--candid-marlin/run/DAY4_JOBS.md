# Day-4 batch (2026-06-11) — review-gap closure + overnight for final conclusions

Goal: by the morning of 06-12, everything needed for final conclusions — robust geometry,
steering incl. unprompted colours, completed ablations. Pull: `bash run/pull_results.sh`,
then offline: `analyze_geometry.py` (all concepts, + `_core` dumps), `color_ceilings.py` rescore.

## Completed during the day (12512–12516, 12526–12529)

| Job | What | Outcome |
|---|---|---|
| 12512 G1 | per-sample geometry dumps, weekdays L19/23/27/31 + months L23/31 | DONE → Step 11 (two-regime geometry) |
| 12513 S1 | steering head-to-head ×5 methods, weekdays 4 layers + months 2 + generation check | DONE → Step 9 (subspace beats manifold/linear 2–5×; best layer L23/27; no model failure) |
| 12514 A1 | ablation suite (first attempt) | TIME LIMIT after Tue/Thu × L31/L23 (each recovery ≈60–90 min) |
| 12515 H1 | hues12: capture+map+steer | TIME LIMIT after steering (capture 47/82, wheel sweeps in hue order) |
| 12516 I1 | gradient inverse, weekdays L23/31 + months L31 | DONE → Step 10 (interior residual 0.20→0.065; retraction) |
| 12526 A2a | weekday ablations: Sun, {Sat,Sun}, {Mon,Tue,Wed}, Mon/Wed titration 25/50/75% | running |
| 12527 A2b | months Feb–Dec recovery k=13; colour family-exclusion (BlueFam 8 / RedFam 7) | running |
| 12528 H2 | hues12 sparse (RGB / RYB), recovery (Green/Blue), geometry dump | DONE |
| 12529 G2 | weekday CORE-regime dump (Gaussian around anchors, L23/31) | DONE |

## Overnight (12530–12532) — results land by morning

| Job | What | Key outputs |
|---|---|---|
| **12530 N1** | **steer to UNPROMPTED colours** — chart from kept anchors only (colors RGB @L27/31, keep-8 @L31; hues12 RGB @L31; weekdays Mon,Thu sanity), linear-inverse init + gradient through frozen model to held-out vertex targets | `steer_unprompted_*.json` (P_linear vs P_gradient per held token — read with ceilings from `color_ceilings.json`) |
| **12531 N2** | geometry robustness: CORE dumps months L23/31 + hues12 L23/31; weekday box dump seed-1 replicate | `geometry_core_L*.npz`, `geometry_seed1_L*.npz` |
| **12532 N3** | steer_compare completion: months L19+L27; hues12 L23/31 (populated-cycle guard) | `steer_compare_months_L19_27.json`, `steer_compare_hues12.json` |

## Morning checklist (in order)

1. `bash run/pull_results.sh`
2. `python3 code/analyses/simplex_coverage/analyze_geometry.py --concept weekdays --layers 19,23,27,31`
   (now also picks up `_core`/`_seed1` dumps — core completes the 0–2 NN chart curve), then months, hues12.
3. Read `steer_unprompted_*.json` against `color_ceilings.json` → the "steer colours never prompted" table.
4. A2a/A2b recovery JSONs → ablation tables (weekday 7/7 + multi-region + titration curves; months 12/12; colour families).
5. `steer_compare_months_L19_27.json` + `steer_compare_hues12.json` → full cross-layer E_BC tables; paired
   significance offline from per_pair lists (scipy wilcoxon).
6. Fill the TODO numbers in `result/CONCLUSIONS.md`, finalize.
