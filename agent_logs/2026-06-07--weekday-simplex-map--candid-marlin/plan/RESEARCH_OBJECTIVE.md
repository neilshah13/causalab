# Research Objective — weekday-simplex-map

**Session:** `agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/`

---

## Objective

Characterize the map between the **behavioural space** for the weekday concept (the region of the
probability simplex Δ^7 over {Mon…Sun} where weekday mass is high and "other" mass is low) and the
**activation space** (residual-stream activations) of Llama-3.1-8B — and, as the first milestone,
establish whether a deliberately diverse, hand-designed set of weekday prompts produces next-token
distributions that **cover that behavioural simplex densely and diversely enough** to make
downstream PCA / manifold fitting (and the behaviour↔activation map) well-conditioned.

## Motivation

The repo originates from *"Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior"* (Wurgaft, Rager, Kowal et al.), which fits an activation manifold
**M_h** and a behavior manifold **M_y** (output distributions on the simplex) and shows that
steering along **M_h** produces behaviours that follow **M_y**, while linear steering does not. The
paper's weekday case, however, drives the simplex using **only** the arithmetic family
("What day is *k* days after *z*?"), whose distributions sit near the simplex vertices (the answer
day) with a little spread onto cyclically-adjacent days. To study the behaviour↔activation **map**
itself — rather than just steer between a handful of vertex-like belief states — we need many belief
states scattered across the *interior, edges, and faces* of the simplex. A prompt set engineered to
elicit such varied distributions (e.g. negation → faces, first-letter constraints → edges,
"name a day" → interior, successor relations → near-vertex-with-neighbour-spread) would give a
much richer, better-conditioned sample of M_y, which is the precondition for everything downstream
(fitting M_y, pairing with M_h, and probing the map). This first step decides how many prompts we
need, builds them, and validates their simplex coverage empirically on the Llama model.

## Scope boundaries

This session's investigation is the behaviour↔activation map for weekdays; **the immediate
deliverable (this first cinaps run) is the prompt set + a coverage validation only.** Explicitly
**not** in scope for this first step:

- **No activation capture / M_h fitting yet.** This step measures *output distributions* only
  (behaviour space). Residual-stream capture for the same prompts is the next step, designed so the
  prompt set is reusable for it.
- **No manifold fitting, steering, or pullback yet** (`output_manifold`, `activation_manifold`,
  `path_steering`, `pullback`). Those come once coverage is validated.
- **No other concept domains** (months, hours, ages, alphabet) — weekdays only.
- **No multilingual prompts** (the FR/ES presets are out of scope).
- **No model training / fine-tuning.** Inference only.
- **Model scope:** a single Llama-3.1-8B checkpoint (base vs instruct decided empirically in the
  debug pass — see flags). 70B is out of scope for this step.

## Success criteria *(recommended)*

Coverage of the behavioural simplex by the retained prompt set (after applying the mass filter
"weekday mass > 90% **and** other mass < 10%"):

- **Yield.** ≥ ~200 prompts pass the mass filter (target band 200–300 designed; expect some
  attrition). If far fewer pass, the format/model choice needs revisiting.
- **Diversity, not collapse.** The retained distributions do **not** collapse onto a few
  vertices: Hellinger-PCA on the retained points needs ≥ 5 of the 7 simplex dimensions to reach
  ~90% explained variance (i.e. effective rank near the simplex dimension), rather than 2–3.
- **Vertex + interior coverage.** All 7 days appear as the arg-max day for ≥ a handful of prompts
  each (no day unreachable), AND a substantial fraction of prompts are genuinely interior
  (e.g. ≥ 25% of retained points have entropy ≥ half of log 7, i.e. not near-one-hot).
- **Spread.** The pairwise-Hellinger distribution among retained points spans a wide range (covers
  near-0 up to near-maximal separations), and a 2-D simplex projection shows points filling the
  region rather than sitting in a few tight clusters.

(These are coverage criteria for *step 1*. Map-quality criteria — isometry / steering fidelity —
will be set when the map-fitting steps are planned.)

## Hypotheses *(recommended)*

- **H1 — families place mass where intended.** Each prompt family lands in a predictable simplex
  region: negation ("not Tuesday") → the face excluding that day; first-letter ("starts with T")
  → the {Tue,Thu} edge (and "S" → {Sat,Sun}); "name a day of the week" → interior / a typical-day
  bias; successor/predecessor → near the answer vertex with adjacent-day spread. *Falsified if*
  families are indistinguishable in simplex coordinates (all collapse to one region).
- **H2 — the chosen model/format keeps mass on weekdays.** With the chosen prompt format, the
  large majority of prompts put > 90% next-token mass on the 7 weekday tokens and < 10% on
  "other". *Falsified if* a large fraction leak > 10% to off-concept tokens (would force an
  instruct model, a stronger completion frame, or few-shot priming).
- **H3 — cyclic neighbour spread (paper-aligned).** For successor/arithmetic prompts, residual
  mass concentrates on **cyclically adjacent** days (Friday answer → Thu/Sat), reproducing the
  paper's §2.1 hypothesis. *Falsified if* residual mass goes to non-adjacent days or there is no
  systematic neighbour structure.

---

## Open decisions / flags (to confirm at the plan checkpoint)

1. **Base vs instruct model + prompt format.** Instruction-style prompts ("Name a day that is not
   Tuesday") are followed zero-shot by **llama31_8b_instruct** (chat template) but not necessarily
   by the **base llama31_8b**, which needs a completion frame ("Q: …\nA:" or "A day of the week
   that is not Tuesday:"). The manifold-steering work uses the base model, so for downstream
   continuity I lean **base + Q/A completion frame**, but propose to **decide empirically in the
   debug pass** (test a small sample on both, compare weekday-mass and instruction-following).
2. **Prompt count = ~250 (band 200–300).** Justified by comparison to past code: the existing
   weekdays runs use `enumerate_all` over 7 entities × 7 numbers = **49 unique prompts** clustered
   at vertices, and subspace PCA uses `k_features ≈ 8–16` (~2–3× the 7 classes). To span the full
   7-dim simplex *and* support later PCA on 4096-dim activations, ~250 diverse points (≈5× the old
   49, ≈30–40× the simplex dimension) is a comfortable, not-wasteful target.
3. **"Other"/weekday tokenization.** Mass is summed over the canonical single tokens
   " Monday"…" Sunday"; leakage to case/format variants (" monday", "Monday" w/o space) is
   reported as a diagnostic. To be confirmed against the Llama-3.1 tokenizer on cinaps.
4. **Forking from the existing pipeline.** The shipped `output_manifold`/`baseline` analyses assume
   a per-class causal-model structure (each example has one true class); our distribution-valued
   prompts have no single "class," so step 1 uses a small **custom coverage analysis** rather than
   those. Long-term reuse of `output_manifold` for M_y fitting will need adaptation — noted for
   later.
