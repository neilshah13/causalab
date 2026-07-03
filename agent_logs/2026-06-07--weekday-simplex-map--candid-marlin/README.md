# 2026-06-07--weekday-simplex-map--candid-marlin

This session investigates the **map between behavioural space and activation space** for the
weekday concept in a Llama model. Behavioural space is the probability simplex over the seven
weekday tokens (regions where the summed weekday probability mass is high — >90% — and all
non-weekday mass is low — <10%). The goal is to learn/characterise how points in this output
simplex correspond to points in the model's residual-stream activation space. **Step 1 (this
first push):** design a fresh, diverse prompt set that scatters many points across the relevant
sub-region of the weekday simplex (so that downstream PCA over both the simplex and the
activations is well-conditioned), then validate the coverage by running the prompts through the
Llama model on the Cinaps cluster and inspecting the resulting distribution of weekday
probability vectors. Deliverable for this step: a validated prompt set + a coverage report.
This is a long, multi-run investigation; later steps will fit and probe the behavioural↔activation
map itself.

## Layout

- `plan/` — research objective, task-spec drafts, approval-checkpoint logs
- `run/` — resolved-config snapshot (`--cfg job` output), `run.log`, slurm logs
- `result/` — `REPORT.md` (single consolidated interpretation written by `/interpret-experiment`), `figures/` for embedded plots/tables
- `code/` — session-local Python + Hydra (via `/setup-methods`, `/setup-analyses`, `/run-experiment`)
- `artifacts/` — raw experiment outputs at `{task}/{model}/{analysis}/...`
- `issues.md` — top-level issue log spanning all phases (managed by `/document-issues`)
