# Cinaps launcher (`deploy/cinaps/`)

Self-contained kit to run `causalab` jobs on the IMO **Cinaps** SLURM cluster.
Cluster facts (SSH, GPU nodes, `/workdir2`, no-QOS) live in
[`.claude/skills/running-on-cinaps/SKILL.md`](../../.claude/skills/running-on-cinaps/SKILL.md).

## Files

| File | Role |
|---|---|
| `config.sh` | shared vars (workdir base, repo URL/branch, models, gate tokens). Override via env. |
| `bootstrap.sh` | one-time login-node setup: install `uv`, `uv sync`, pre-download weights. |
| `pilots.tsv` | the manifest — one job per line (`name <tab> gpus <tab> command`). |
| `launch_all.sh` | submit every manifest line to the queue (one `sbatch` each). |
| `job.sbatch` | generic job step — sets offline HF env, runs `uv run <command>`. |
| `monitor.sh` | `squeue`/`sacct` + recent logs. |
| `encoding_gate.py` | tokenization-robust Stage-0 gate (generation + lenient first-word match). |

## Flow (three commands)

```bash
# 1. clone + bootstrap (once, on the login node — it has internet)
ssh -F ~/.ssh/cinaps_ssh_config cinaps
git clone --branch cinaps-launchers https://github.com/neilshah13/causalab.git /workdir2/$USER/causalab
cd /workdir2/$USER/causalab
HF_TOKEN=hf_xxx bash deploy/cinaps/bootstrap.sh

# 2. launch the whole manifest
HF_HOME=/workdir2/$USER/hf_cache bash deploy/cinaps/launch_all.sh

# 3. watch
bash deploy/cinaps/monitor.sh
```

## What it runs now

`pilots.tsv` holds the **Phase C encoding gate** for the four multilingual
cycles (FR/ES weekdays + months). Each line runs `encoding_gate.py`, which
generates a few tokens and does a lenient first-word match — so multi-token
answers (e.g. `mercredi → [' mer','cre','di']`) are scored on knowledge, not
tokenization. Results land at:

```
artifacts/natural_domains_arithmetic/llama31_8b/<variant>/encoding_gate/accuracy.json
```

`accuracy.json` reports `accuracy`, `accuracy_accent_insensitive`,
`encoding_gate_pass` (≥ 0.60), and a `per_class` breakdown; `examples.json`
holds sample prompt→generation pairs for eyeballing.

## Adding work

Append lines to `pilots.tsv`. Any `uv run …` command works — a gate
(`encoding_gate.py …`) or a full pipeline (`-m causalab.runner.run_exp
--config-name <runner>`). Long-context pipelines (graph_walk) need ≥ 48 GB:
add `--nodelist=node11,node22` via a per-line tweak or submit those by hand.
