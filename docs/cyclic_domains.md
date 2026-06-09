# Cyclic concept domains — task catalog & runner recipes

This branch adds **20 task presets** to `natural_domains_arithmetic`, extending the original calendar cycles (weekdays + months) into multilingual variants and additional English cyclic concept domains. Each preset is a drop-in replacement for the existing `/task` Hydra group — same causal model, same counterfactual mechanism, only the entities, numbers, and prompt template change.

Use this doc to:

- Pick a cycle to run the manifold steering pipeline on.
- Understand which cycles a given base model can actually represent (encoding gate).
- Write a runner config for a cycle that doesn't have one yet.

## Catalog

### 1. Multilingual calendar cycles

Cross-language transfer experiments for the original weekdays/months cycles.

#### European (FR/ES)

| Task name | Entities | Modulus | Template |
|---|---|---|---|
| `natural_domains_arithmetic_weekdays_fr` | lundi … dimanche | 7 | `Quel jour est {number} jours après {entity}?` |
| `natural_domains_arithmetic_weekdays_es` | lunes … domingo | 7 | `¿Qué día es {number} días después de {entity}?` |
| `natural_domains_arithmetic_months_fr` | janvier … décembre | 12 | `Quel mois est {number} mois après {entity}?` |
| `natural_domains_arithmetic_months_es` | enero … diciembre | 12 | `¿Qué mes es {number} meses después de {entity}?` |

#### CJK & Hindi

Offsets use Arabic numerals (1–7) in all three languages — avoids Chinese 两/二 ambiguity
and simplifies tokenization. `max_new_tokens=10` (entities are multi-token in Llama's tokenizer).

| Task name | Entities | Modulus | Template |
|---|---|---|---|
| `natural_domains_arithmetic_weekdays_zh` | 星期一 … 星期日 | 7 | `{entity}后{number}天是哪天？` |
| `natural_domains_arithmetic_months_zh` | 一月 … 十二月 | 12 | `{entity}后{number}个月是哪个月？` |
| `natural_domains_arithmetic_weekdays_ja` | 月曜日 … 日曜日 | 7 | `{entity}の{number}日後は何曜日ですか？` |
| `natural_domains_arithmetic_months_ja` | 1月 … 12月 | 12 | `{entity}の{number}ヶ月後は何月ですか？` |
| `natural_domains_arithmetic_weekdays_hi` | सोमवार … रविवार | 7 | `{entity} के {number} दिन बाद कौन सा दिन होगा?` |
| `natural_domains_arithmetic_months_hi` | जनवरी … दिसंबर | 12 | `{entity} के {number} महीने बाद कौन सा महीना होगा?` |

### 2. Additional English cyclic domains

Cycles beyond the calendar: spatial, musical, cultural.

| Task name | Entities | Modulus | Notes |
|---|---|---|---|
| `natural_domains_arithmetic_moon_phases` | New Moon, First Quarter, Full Moon, Last Quarter | 4 | 4-quarter version; the 8-phase variant collides on first tokens under `max_new_tokens=1` |
| `natural_domains_arithmetic_solfege` | Do, Re, Mi, Fa, Sol, La, Ti | 7 | same modulus as weekdays |
| `natural_domains_arithmetic_compass` | North, East, South, West | 4 | |
| `natural_domains_arithmetic_zodiac` | Aries … Pisces | 12 | same modulus as months |
| `natural_domains_arithmetic_chinese_zodiac` | Rat … Pig | 12 | same modulus as months |

### 3. Few-shot variants

For each of the 5 non-calendar cycles, a `_fs` variant prefixes the test prompt with 2 worked Q/A demonstrations (one within-cycle, one wrap-around). Mechanism, entities, modulus unchanged; only the template grows.

| Task name | Source cycle |
|---|---|
| `natural_domains_arithmetic_moon_phases_fs` | moon_phases |
| `natural_domains_arithmetic_solfege_fs` | solfege |
| `natural_domains_arithmetic_compass_fs` | compass |
| `natural_domains_arithmetic_zodiac_fs` | zodiac |
| `natural_domains_arithmetic_chinese_zodiac_fs` | chinese_zodiac |

## Running a cycle through the pipeline

The new task configs are drop-in replacements at the `/task` Hydra group. Take the canonical reference runner `causalab/configs/runners/weekdays/weekdays_8b_pipeline.yaml` and swap two lines:

```yaml
# @package _global_
defaults:
- /base
- /task: natural_domains_arithmetic_zodiac   # ← swap this for any cycle name above
- /model: llama31_8b                          # ← swap for the model you want
- /analysis/baseline
- /analysis/subspace
- /analysis/activation_manifold
- /analysis/output_manifold
- /analysis/path_steering
- /analysis/pullback                          # drop for cycles with ≥12 entities (too expensive)
- _self_

task:
  target_variable: result

subspace:
  layers: [28]                                # ← layer choice depends on model
activation_manifold:
  layers: [28]

path_steering:
  n_extra_pairs: 29
  isometry:
    n_interior_per_pair: 4
```

Save under `causalab/configs/runners/<cycle>/<cycle>_<model>_pipeline.yaml` and run:

```bash
./scripts/run_exp.sh <runner_name>            # inline
./scripts/run_exp.sh --slurm <runner_name>    # sbatch
```

### Stage-0 encoding-gate sweep only

If you only want to know whether a cycle passes the 60% baseline-accuracy encoding gate (before paying the cost of subspace/manifold/path_steering), drop everything but `baseline`:

```yaml
defaults:
- /base
- /task: natural_domains_arithmetic_zodiac
- /model: llama31_8b
- /analysis/baseline
- _self_

task:
  target_variable: result
```

This runs in a few seconds per cycle on a single GPU.

### Layer choice per model

The proposal's "2/3 down the stack" heuristic was confirmed on Llama-3.1-8B (L28 of 32 = 87.5% depth) but is too shallow on Gemma 2 and Gemma 4:

| Model | num_hidden_layers | Ring-forming layer | Depth |
|---|---|---|---|
| Llama-3.1-8B base | 32 | L28 | 87.5% (≈ 2/3 from proposal) |
| Gemma 2 9B | 42 | L41 | 97.6% (very late) |
| Gemma 4 31B base | 60 | L50 | 83% (range L40–L58 all score 1.0 on PCA; L50 picked as middle) |

If you're trying a new model, do a layer sweep first: set `subspace.layers: [L_2/3, L_3/4, L_late_minus_2]` (skipping `activation_manifold`/`output_manifold`/`path_steering` to keep it cheap), inspect the saved 3D scatter plots, then pin a single layer for the full pipeline.

## Encoding-gate findings

Pass/fail at the 60% top-1 baseline accuracy threshold across model × cycle. Pass = the model produces the correct next token as its argmax on ≥ 60% of (entity, number) pairs.

### English non-calendar cycles

| Cycle | Llama-3.1-8B base | Gemma 2 9B base | Gemma 4 31B base | Gemma 4 31B IT (via API) |
|---|---|---|---|---|
| weekdays | **94%** ✓ | **92%** ✓ | **96%** ✓ | 100% ✓ |
| months | **100%** ✓ | **100%** ✓ | **100%** ✓ | 100% ✓ |
| moon_phases | 0% | 0% | — | 25% |
| solfege | 14% | 27% | — | 43% |
| compass | 25% | 31% | — | 58% (4-cycle), 32% (8-cycle) |
| zodiac | 4% | 42% | — | **90%** ✓ |
| chinese_zodiac | 13% | 17% | — | **80%** ✓ |
| hours_24 | — | — | — | **88%** ✓ |
| chromatic | — | — | — | **79%** ✓ |

The small base models pass only the calendar cycles. Gemma 4 31B IT via chat-template wrapping passes 7 of 9 — adding music, 24-hour clock, and both zodiacs to the geometric-encoding club. The `_fs` few-shot variants lift some failing cycles into the gate window (e.g., chinese_zodiac on Qwen 3.5 9B: 4% → 48%).

### Multilingual calendar cycles — European (Gemma 4 31B base, raw Q/A)

| Cycle | EN | FR | ES |
|---|---|---|---|
| weekdays | 96% | **47%** ✗ | 82% |
| months | 100% | **52%** ✗ | (in flight) |

The FR base-model accuracy drops below the gate despite the IT variant via OpenRouter producing 100% on the same prompts — a chat-template effect. **Crucially, the underlying activation manifold at L50 is still cyclic on French weekdays (isometry r = 0.984)** even though the model fails to output the right token most of the time. Encoding ≠ readout.

### Multilingual calendar cycles — CJK & Hindi (results pending)

Encoding-gate sweep to be run on Llama-3.1-8B via `encoding_gate.py` (lenient first-word
matching; exact-match `baseline` accuracy will be near 0% for Hindi due to Devanagari
tokenization). The `—` entries below will be filled after Phase 1 runs.

| Cycle | Llama-3.1-8B (lenient) |
|---|---|
| weekdays_zh | — |
| months_zh | — |
| weekdays_ja | — |
| months_ja | — |
| weekdays_hi | — |
| months_hi | — |

Runner configs: `causalab/configs/runners/multilingual/*_zh_*.yaml`, `*_ja_*.yaml`, `*_hi_*.yaml`.
Gate runner configs (baseline only): session-local `*_gate.yaml` files in
`agent_logs/2026-06-10--cjk-hindi-cycles--keen-panda/code/configs/runners/multilingual/`.

## Caveats

- **Instruct (`-it`) vs base models on raw `Q:/A:` templates.** The shipped task templates are formatted for *base* models that pattern-complete on plain Q/A text. Instruct models often need a chat-template wrapper to behave correctly on the same prompts. The encoding gate measured on a raw `Q:/A:` template against an `-it` model will under-report (see Gemma 3 27B IT → French weekdays in the report). If you want to compare instruct-model encoding, either use the base counterpart or wire `use_chat_template=True` through the pipeline.
- **Pullback is expensive.** The pullback analysis (LBFGS belief-space optimization) costs ~5 min per pair × n_pairs. For cycles with ≥ 12 entities (`months`, `zodiac`, `chinese_zodiac`, `chromatic`), n_pairs = 50 → ~4 h. Drop pullback from the runner unless you need belief-space visualizations.
- **First-token matching.** `max_new_tokens=1` means the model must produce the right answer as its very first token. Tasks with multi-word entities that share leading tokens (`moon_phases`'s 8-phase variant: "Waxing Crescent" / "Waxing Gibbous" both start with "Waxing") will collapse to single-token accuracy unrelated to whether the model knows the cycle. The 4-quarter `moon_phases` preset sidesteps this; if you add a new cycle with multi-word entities, verify first-token uniqueness before scoring.

## See also

- [`README.md`](../README.md) — the canonical "run the headline weekdays pipeline" path.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) §2 — full Hydra config-group mechanics.
- [`causalab/tasks/natural_domains_arithmetic/config.py`](../causalab/tasks/natural_domains_arithmetic/config.py) — `DOMAIN_PRESETS` source of truth for entities, modulus, templates.
- [`agent_logs/2026-05-30--manifold-cycles--lucid-cedar/result/REPORT.md`](../agent_logs/2026-05-30--manifold-cycles--lucid-cedar/result/REPORT.md) — the cross-model study these encoding-gate numbers come from.
