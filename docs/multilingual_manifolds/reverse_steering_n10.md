# Reverse (X→EN) cross-source steering — raising n to 10 (+ significance)

**Date:** 2026-06-16 · **Model:** Gemma-3-27B base @ L54, last-token, pca_k64 · **Target:** English weekday ring.
**Follow-up to** `agent_logs/2026-06-15--belief-rescore-steering--vivid-marlin` (Exp B, n=6).

## What ran

Four new source→EN steering runs added to close the "raise n + add a significance test" item from
PRESENTATION.md Part VI. SLURM job `12754` (node11), runners
`weekdays_{es,hi,id,tr}_en_steering_gemma`, target tree `…/gemma3_27b/weekdays`.

New transfer coherence (mean ± SE): **es→EN 0.733 ± 0.030 · hi→EN 0.839 ± 0.003 · id→EN 0.701 ± 0.017 · tr→EN 0.785 ± 0.008.**

## Full n=10 table (uniform, computed from artifacts)

`overlap` = mean cos² principal angles between source and EN pca_k64 subspaces (sanity: overlap(EN,EN)=1.0,
k/D=0.0089). `own_coh` = source path_steering geometric coherence (belief-scoring-invariant). `isometry` =
source path_steering geometric pearson_r (**position-scored**, uniform across all 10 — see caveat).

| src | new | transfer→EN | overlap | own_coh | isometry |
|---|---|---|---|---|---|
| vi |   | 0.708 | 0.095 | 0.676 |  0.081 |
| sw |   | 0.828 | 0.105 | 0.665 |  0.221 |
| ja |   | 0.862 | 0.220 | 0.394 |  0.081 |
| fr |   | 0.811 | 0.295 | 0.967 |  0.443 |
| zh |   | 0.793 | 0.193 | 0.965 | −0.108 |
| ko |   | 0.808 | 0.184 | 0.989 |  nan |
| es | * | 0.733 | 0.292 | 0.609 |  0.410 |
| hi | * | 0.839 | 0.250 | 0.916 |  0.102 |
| id | * | 0.701 | 0.206 | 0.511 |  0.196 |
| tr | * | 0.785 | 0.104 | 0.658 |  0.089 |

## Correlations (Pearson r, p)

| predictor | n=6 (prior) | **n=10 (now)** |
|---|---|---|
| transfer ~ overlap   | r=+0.488, p=0.326 | **r=+0.138, p=0.704** |
| transfer ~ own_coh   | r=−0.244, p=0.641 | **r=+0.231, p=0.521** |
| transfer ~ isometry  | r=+0.190, p=0.760 (n=5) | **r=−0.143, p=0.714 (n=9)** |

(n=6 overlap/own_coh reproduce vivid-marlin's 0.489 / −0.244 exactly → pipeline validated.)

## Conclusion

**At n=10, none of subspace overlap, source own-coherence, or source isometry predicts X→EN transfer
(all |r|≤0.23, all p>0.5).** This *strengthens* the vivid-marlin refutation of the source-quality law for
reverse transfer: the n=6 overlap correlation (r≈0.49) was never significant (p=0.33) and **collapses to
r≈0.14 with four more points** — it was small-sample noise, not signal. No source ring-quality proxy carries
predictive weight either.

The dominant, robust effect is the **EN→X ≫ X→EN asymmetry**: ten typologically diverse sources (clean
Latin rings es/fr, broken CJK ja/zh, Devanagari hi, low-resource vi/sw/id/tr) all steer English into a
narrow **0.70–0.86** band, versus EN-as-source forward transfer of ~0.95. Transfer *to* English is bounded
by English's own target-steerability ceiling and is essentially insensitive to which source ring is used —
consistent with a largely language-agnostic "cyclic successor" operation reading into a near-constant EN
readout, with source-specific ring quality washing out in the reverse direction.

### Caveats
- **Isometry provenance:** values above are position-scored (original roots), uniform across all 10 but
  re-introducing the KO `nan` / negative-ZH artifacts that vivid-marlin's `answer_sequence` re-score fixed.
  `own_coh` (belief-invariant) is the cleaner ring-quality axis and is also null. A same-provenance
  `answer_sequence` isometry for all 10 (vi/sw/es/hi/id/tr not yet re-scored) would tighten the isometry row,
  but cannot rescue a law that own-coherence already shows is absent.
- n=10 is still modest; the claim is "no predictor reaches significance and the only prior positive
  (overlap) is not robust," not a tight null CI.
