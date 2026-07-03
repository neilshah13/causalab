# PLAN — weekday-simplex-map (Step 1: prompt set + simplex-coverage validation)

Sibling: [`RESEARCH_OBJECTIVE.md`](RESEARCH_OBJECTIVE.md). This plan covers **only the first cinaps
run**: build a diverse weekday prompt set and validate how well its next-token distributions cover
the behavioural simplex. Downstream map-fitting steps (M_y, M_h, steering) are out of scope here.

---

## §B. Prompt set & "behaviour" definition

> NOTE: This step does **not** use the `natural_domains_arithmetic` causal-model/counterfactual
> machinery. The shipped `baseline`/`output_manifold` analyses assume each example has one true
> *class* (`intervention_value_index`); our prompts are **distribution-valued by design** (a prompt
> like "name a day that is not Tuesday" has no single correct day). So step 1 defines a flat,
> annotated **prompt set** (a session-local data module) consumed by a small **custom** coverage
> analysis. No causal variables, no mechanisms, no counterfactuals.

**Artifact:** `${SESSION_DIR}/code/methods/weekday_prompts/prompts.py` — programmatically builds the
prompt list and emits `prompts.json`. Each record:

```json
{"id": "...", "family": "negation", "text_core": "name a day of the week that is not Tuesday",
 "intended_days": ["Mon","Wed","Thu","Fri","Sat","Sun"], "paraphrase_id": 0}
```

`intended_days` is a **diagnostic annotation only** (used to color plots and sanity-check H1), never
a hard label. `text_core` is model-agnostic; the model/format wrapper (Q/A vs chat) is applied at
run time so the same set serves the "test both" comparison.

### Domain

`Z = {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}` (7 days). Behaviour point for
a prompt `x`: `p(x) ∈ Δ^8` = `[P(" Monday"), …, P(" Sunday"), P(other)]`, where each weekday
probability is the next-token (first generated token) probability of that day's token and
`P(other) = 1 − Σ_days`. **Mass filter for "covers the simplex":** keep `x` iff
`Σ_days P ≥ 0.90` and `P(other) ≤ 0.10`.

### Prompt families (target ≈ 250 after paraphrase expansion & dedup)

Designed so different families land in different regions of the simplex (H1). Counts are post-
expansion targets; the builder expands `template × {day|letter|offset} × paraphrase` then dedups.

| # | Family | Simplex region it targets | Construction | ~Count |
|---|---|---|---|---|
| A | **open "name a day"** | interior / typical-day bias | ~10 paraphrases of "name a day of the week" (name/tell/pick/choose/give/say/random…) | ~10 |
| B | **single negation** "not X" | a 6-day **face** (excludes X) | 7 days × ~3 paraphrases (not / other than / except / any day but) | ~21 |
| C | **double negation** "not X and not Y" | 5-day face / interior | all C(7,2)=21 pairs × 1 (subsample if needed) | ~21 |
| D | **first-letter** "starts with L" + converse | **edges** ({Tue,Thu}, {Sat,Sun}) & vertices | letters {M,T,W,F,S} for "starts with"; same for "does not start with"; ×2 paraphrases | ~20 |
| E | **successor / predecessor / arithmetic** | near-vertex + **cyclic neighbour spread** (paper-aligned, H3) | "{day after / day before / two days after / three days after} {X}" and "Q: what day is {k} days after {X}?\nA:" for k∈{1,2,3}; 7 days each, dedup | ~50 |
| F | **semantic constraint** | edges / specific vertices / bimodal | weekend day; weekday/work day/school day; first day of week (bimodal Mon/Sun); last day; middle of week; day before the weekend; first day of the weekend; … | ~25 |
| G | **ordinal / positional** | spread by Mon-start vs Sun-start convention | "the {1st..7th} day of the week is" (+ "weekday" variant) | ~14 |
| H | **graded / hedged & "today/tomorrow"** | interior, soft distributions | "a day near the end of the week", "most likely the busiest day", "if today is X, tomorrow is", "the day after tomorrow if today is X" | ~25 |
| I | **set-membership / multiple constraints** | small faces & edges | "a weekend day that is not Sunday" (→Sat), "a weekday that starts with T" (→Tue,Thu), "a day between Tuesday and Friday", … | ~25 |

Rough sum ≈ 211 distinct; paraphrase multiplicity in A/B/D and a few extra arithmetic offsets in E
push to ~250. **If natural-language distinctness caps the truly-distinct regions below ~250, that is
itself a reportable finding** (we'd note that the simplex's naturally-reachable region is smaller
than the design target) rather than padding with near-duplicates.

### Representative concrete prompts (core text, pre-wrapper)

```
A  name a day of the week
B  name a day of the week that is not Tuesday        intended: all but Tue
C  name a day of the week that is neither Tuesday nor Friday   intended: 5 days
D  name a day of the week that starts with the letter T        intended: Tue,Thu
D  name a day of the week that does not start with S           intended: Mon..Fri
E  the day after Tuesday is                                     intended: Wed (+Tue/Thu spread)
F  name a day of the weekend                                    intended: Sat,Sun
F  the first day of the week is                                 intended: Mon | Sun (bimodal)
G  the third day of the week is                                 intended: Wed | Tue
H  if today is Friday, tomorrow is                              intended: Sat
I  a weekend day that is not Sunday is                          intended: Sat
```

### Model/format wrappers (applied at run time)

- **Base** `llama31_8b`: `f"Q: {Text_core.capitalize()}.\nA:"` (completion frame; matches repo
  conventions). Read logits at the first generated token (`max_new_tokens=1`).
- **Instruct** `llama31_8b_instruct`: chat template via `pipeline.load(..., use_chat_template=True)`
  with the core text as the user turn; read first generated token.

---

## §C. Neural surface

### Models

| Model | Config | Role |
|---|---|---|
| `meta-llama/Llama-3.1-8B` | `model: llama31_8b` (base) | candidate for full run; Q/A completion frame |
| `meta-llama/Llama-3.1-8B-Instruct` | `model: llama31_8b_instruct` | candidate for full run; chat template |

Debug pass runs a ~24-prompt representative subset on **both**; the winner (higher fraction passing
the mass filter + correct instruction-following on spot-checks) runs the full ~250.

### Tokenization checks (run first, on cinaps login node or in-job)

- Confirm each of `" Monday"…" Sunday"` encodes to a **single** token id under the Llama-3.1
  tokenizer (`add_special_tokens=False`). If any day is multi-token, fall back to first-subtoken
  probability and flag it (mass slightly under-counted for that day).
- Diagnostic: also tokenize case/space variants (`"Monday"`, `" monday"`) and report how much
  next-token mass leaks to them (informs whether the 7 canonical tokens under-count weekday mass).

### Compute budget

| Phase | Where | Wall time | GPUs |
|---|---|---|---|
| tokenizer check | login node (CPU) or in-job | < 1 min | 0–1 |
| debug pass (≈24 prompts × 2 models) | slurm | ~3–5 min (model load dominates) | 1 |
| full run (≈250 prompts × 1 model, max_new_tokens=1) | slurm | ~3–6 min | 1 |

Forward passes are trivial (250 single-token generations); wall time is dominated by loading the
8B checkpoint (~1–2 min, bf16 ~16 GB). One GPU on any cinaps node suffices.

### Hardware constraints

- **Cluster (cinaps):** SLURM, `--gpus=1`, no `--qos`/`--partition`/`--account`. Work under
  `/workdir2/<login>`. Pre-download HF weights on the login node; run with `HF_HUB_OFFLINE=1`. See
  `.claude/skills/running-on-cinaps/SKILL.md` and the `cinaps-access` memory.
- Llama-3.1-8B (base + instruct) both fit comfortably on the smallest (24 GB) node.

---

## §D. Analysis-chain DAG

```
weekday_prompts (data)  ──►  format_probe (debug, both models)  ──►  pick model
                                                                       │
weekday_prompts (data)  ─────────────────────────────────────────────►  simplex_coverage (full)
```

Both nodes are **custom / session-local** (no shipped analysis fits distribution-valued prompts).
Implementation: a single session-local script invoked twice (probe mode, then full mode), reusing
`causalab.io.pipelines.load_pipeline` + `LMPipeline.generate`. It is **not** wired through the Hydra
task/analysis runner (the runner requires a causal-model task). Artifacts still land under the
session's `artifacts/` tree per conventions.

#### Node 1: `format_probe` (custom — debug)

- **Scoped question:** Which model/format keeps next-token mass on weekdays and follows the trickier
  instructions? (Resolves the "test both" fork.)
- **Process:** run ~24 representative prompts (≥2 per family, incl. the hard ones: negation,
  first-letter, ordinal) on base (Q/A) and instruct (chat); record per-prompt weekday-mass,
  P(other), top-5 decoded next tokens, and whether the arg-max day respects `intended_days`.
- **Downstream artifacts:** `…/simplex_coverage/format_probe.json` (+ a short markdown table).
- **Pre-flight check (gate):** the winning model must yield **weekday-mass ≥ 0.90 on ≥ 60%** of the
  24 probe prompts AND visibly respect constraints (e.g. "starts with T" puts mass on Tue/Thu).
  If neither model clears it → stop, revisit the wrapper (stronger completion frame / few-shot
  priming) before the full run.
- **Runtime:** ~5 min on 1 GPU.
- **Spec:** `${SESSION_DIR}/plan/setup_analysis_simplex_coverage.md` (covers both nodes).

#### Node 2: `simplex_coverage` (custom — full)

- **Scoped question:** Do the ~250 prompts' retained distributions cover the behavioural simplex
  densely and diversely (the step-1 success criteria)?
- **Process:**
  1. Run all prompts (chosen model), `max_new_tokens=1`, batched; softmax the first-token logits.
  2. Build `dists` `(N, 8)` = `[7 weekday probs, other]`; record per-prompt top-10 decoded tokens.
  3. Apply mass filter (≥0.90 weekday, ≤0.10 other) → retained set `R`.
  4. Coverage metrics on `R` (Hellinger geometry, matching the paper's `√p` convention):
     - **yield** overall and per family;
     - **Hellinger-PCA** explained-variance-ratio curve → #dims for 90% var; effective rank;
     - **entropy** distribution → fraction "interior" (entropy ≥ 0.5·log 7);
     - **per-day arg-max coverage** counts (all 7 reachable?);
     - **pairwise Hellinger** distance distribution (min/median/max + histogram);
     - **convex-hull volume** in 2–3 D Hellinger-PCA coords, vs. the hull of the 7 one-hot vertices
       (what fraction of the achievable region is filled).
  5. Plots → `…/figures/`: 2-D Hellinger-PCA scatter colored by (a) family and (b) arg-max day;
     entropy histogram; pairwise-distance histogram; per-family centroid heatmap (7-day bars);
     interactive 3-D HTML scatter.
- **Upstream artifacts consumed:** `prompts.json`; chosen model id from `format_probe.json`.
- **Downstream artifacts produced:** under
  `${SESSION_DIR}/artifacts/weekday_simplex/{model}/simplex_coverage/`:
  `distributions.safetensors`, `prompts_scored.json`, `coverage_metrics.json`, `figures/*`.
- **Non-default knobs:** N/A (custom script; config snapshotted to `run/`).
- **Pre-flight check (gate / = step-1 success criteria):** ≥ ~200 retained; ≥ 5 Hellinger-PCA dims
  for 90% var; all 7 days reachable as arg-max; ≥ 25% interior points. Meeting these = deliverable
  satisfied. Missing them → diagnose (which families collapsed?) and iterate on the prompt set.
- **Runtime:** ~6 min on 1 GPU.

### Cross-analysis post-steps

None (single full node). The coverage report is assembled by `/interpret-experiment` into
`result/REPORT.md`.

### Optional cheap add-on (flagged, not in step-1 scope)

Because the model is already loaded and prompts already run, we **could** in the same pass dump
last-token residual-stream activations (all layers, `(N, n_layers, 4096)` — a few hundred MB) so the
prompt set is paired for the next (M_h) step and we save a cinaps run. Left **out** of step 1 per
the stated scope; trivial to enable if you want it folded in.

---

## §E. Risk register & contingency

### Pitfalls active for this plan

- **Base model ignores instructions** → low weekday-mass / wrong constraint following. *Mitigated*
  by the "test both" probe (instruct fallback) and the `format_probe` gate.
- **Multi-token weekday names** → mass under-count. *Mitigated* by the tokenizer check + first-
  subtoken fallback + leakage diagnostic.
- **Distinctness ceiling** — natural language may reach fewer distinct simplex regions than 250.
  *Handled* by reporting it as a finding, not padding with near-duplicate paraphrases.
- **Coverage bias** — distributions may pile up near a few days (e.g. Monday/typical-day) or near
  uniform. *Handled* by per-day arg-max coverage + per-family centroid heatmap; if a day/region is
  unreachable, add targeted constraints (family I) and re-run (cheap).
- **Not a Hydra task** — deliberate deviation from the standard runner; provenance preserved by
  snapshotting the script + resolved settings into `run/`.

### Per-step contingency

| Node | If pre-flight fails, then |
|---|---|
| `format_probe` | neither model ≥ 60% pass → add a stronger completion frame and/or 1–2 neutral few-shot exemplars (careful not to bias day mass), re-probe. Do not launch the full run. |
| `simplex_coverage` | < ~200 retained or collapsed PCA (≤4 dims) or a day unreachable → identify the failing families, add/adjust prompts (family I targeted constraints), re-run the (cheap) full pass. |

---

## §F. Outputs of the plan itself

### Entry point (not a Hydra runner config)

- `${SESSION_DIR}/code/methods/weekday_prompts/prompts.py` — prompt builder → `prompts.json`.
- `${SESSION_DIR}/code/analyses/simplex_coverage/run_coverage.py` — single script, `--mode
  {probe,full}`, `--model {llama31_8b,llama31_8b_instruct}`, `--experiment-root <session
  artifacts>`. Uses `load_pipeline`. (Scaffolded via `/setup-analyses` after approval, or written
  directly.)
- Submitted to cinaps via an `sbatch` wrapper per `/running-on-cinaps` (not `scripts/run_exp.sh`).

### Sweep & cache strategy

No Hydra sweep. The script is run: (1) probe mode on both models; (2) full mode on the chosen model.
Each writes to a distinct `{model}` subdir, so no overwrite hazard.

### Expected artifact tree

```
${SESSION_DIR}/artifacts/weekday_simplex/
├── llama31_8b/                 (and/or llama31_8b_instruct/)
│   └── simplex_coverage/
│       ├── format_probe.json            # debug-pass comparison (both models)
│       ├── distributions.safetensors    # (N, 8) first-token weekday+other dists
│       ├── prompts_scored.json          # per-prompt: text, family, intended_days,
│       │                                #   weekday_mass, other, top-10 tokens, retained?
│       ├── coverage_metrics.json        # yield, PCA EVR, eff. rank, entropy stats,
│       │                                #   per-day argmax, pairwise-Hellinger, hull volume
│       └── figures/
│           ├── pca_scatter_by_family.pdf
│           ├── pca_scatter_by_argmax_day.pdf
│           ├── entropy_hist.pdf
│           ├── pairwise_hellinger_hist.pdf
│           ├── family_centroid_heatmap.pdf
│           └── simplex_3d.html
```

`run/` will hold: the resolved run settings (model, prompt count, seed, filter thresholds), the
slurm job id(s), and `run.log`.

### Hand-off

- After approval: `/setup-analyses` (or direct authoring) to create the prompt builder + coverage
  script under `${SESSION_DIR}/code/`; then run on cinaps via `/running-on-cinaps`; then
  `/interpret-experiment` to write `result/REPORT.md` (the coverage report + flags).
- Because this is a custom script rather than a Hydra runner, `/run-experiment` is **not** the exact
  vehicle; the cinaps submission is a direct `sbatch` of `run_coverage.py`.

---

## Review checkpoint

1. **Success criteria + hypotheses** (RESEARCH_OBJECTIVE.md): coverage-based (yield ≥200, ≥5 PCA
   dims, all 7 days reachable, ≥25% interior) + H1/H2/H3. Agree these are the right tests?
2. **"Test both" probe then full run** (§C/§D): acceptable to spend one small extra debug pass to
   pick base vs instruct before the full ~250?
3. **Compute** (§C): two short single-GPU jobs (~5–6 min each). Fits your window?
4. **Deviations to confirm:** (a) custom script, not the Hydra runner; (b) activation capture left
   out of step 1 (optional cheap add-on available).
