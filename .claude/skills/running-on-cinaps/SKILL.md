---
name: running-on-cinaps
description: Use when running, submitting, or monitoring experiments on the Cinaps GPU cluster (IMO / Université Paris-Saclay) — SSH access, SLURM sbatch/srun jobs, GPU node selection, or setting up the uv/HuggingFace environment on /workdir2.
---

# Running on the Cinaps cluster

## Overview

Cinaps is the lab's SLURM cluster (IMO / Paris-Saclay). The login node (`cinaps2`) only routes — it has **no GPU and is not for compute**. All work goes through the SLURM queue (`srun` for short/interactive, `sbatch` for real jobs). Facts below are verified against the live cluster, not the public doc.

## Connect

```bash
ssh -F ~/.ssh/cinaps_ssh_config cinaps      # ProxyJump via sas; key auth, no password
```

## Filesystem — use /workdir2, never $HOME

| Path | Use |
|---|---|
| `/workdir2/<login>` | **compute dir** — repo, venv, HF cache, artifacts (35 TB free) |
| `$HOME` | NFS, **not sized for compute** — config only |

`/workdir` does **not** exist on this cluster; use `/workdir2`. (`<login>` = your cluster user, e.g. `johan.boscher`; `whoami`.)

## GPUs — `--gpus=N`, no QOS/partition/account

Single default partition `LMO-CPU`. There is **no `--qos`, `--partition`, or `--account`** — do not pass them. Request GPUs with `--gpus=N`. Schedulable GPU nodes:

| Node | gres | VRAM | Notes |
|---|---|---|---|
| node11 | gpu:1 | RTX PRO 6000, **96 GB** | newest |
| node15 | gpu:1 | GV100, **32 GB** | |
| node20 | gpu:4 | RTX 6000, **24 GB** ×4 | OOM-prone for long-context |
| node22 | gpu:2 | A6000, **48 GB** ×2 | |

8 schedulable GPUs total. node18's old K20m is **not** a SLURM gres, so `--gpus` never lands there — no `--exclude` needed. For **long-context jobs (≥2048 tok, e.g. graph_walk)** that need ≥48 GB, pin to the big cards: add `--nodelist=node11,node22` (or `--exclude=node20,node15`).

## Environment (one-time, on the login node)

- **No system `uv`.** Install the standalone binary into `/workdir2` and run `uv sync`. The repo pins **Python 3.10**; uv fetches its own (system python is miniforge 3.12 — don't use it).
- Only modules present: `module-git`, `singularity/4.3.3`. **No CUDA module** → rely on PyTorch's bundled CUDA wheels (the GPU driver lives on the compute nodes).
- **Login node has internet; treat compute nodes as offline.** Pre-download weights on the login node:
  ```bash
  export HF_HOME=/workdir2/<login>/hf_cache
  export HF_TOKEN=...                          # gated Llama needs a token
  uv run python -m causalab.runner.download_models meta-llama/Llama-3.1-8B
  ```
  Then in every job set `HF_HOME=/workdir2/<login>/hf_cache` and `HF_HUB_OFFLINE=1`.

## Submit a job

`scripts/run_exp.sh --slurm <runner>` works as-is: it resolves `--gres=gpu:N` and `--time` from the Hydra config and only adds `--qos` if you pass `--qos` (so **don't**). Export `HF_HOME`/`HF_HUB_OFFLINE` first (e.g. in `~/.bashrc` on the cluster). Or hand-write the sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=causalab_weekdays
#SBATCH --gpus=1
#SBATCH --chdir=/workdir2/<login>/causalab
#SBATCH --output=slurm_logs/%x_%j.out
export HF_HOME=/workdir2/<login>/hf_cache HF_HUB_OFFLINE=1
uv run python -m causalab.runner.run_exp --config-name weekdays_8b_pipeline
```

For many runs at once, loop the manifest (`for r in ...; do sbatch ... "$r"; done`) or use `sbatch --array` over a manifest file.

## Monitor

`squeue` (running = `R` in ST) · `sacct` (finished-job accounting) · `sattach <jobid>` (live I/O; ctrl-c detaches) · `scancel <jobid>` · `sstat <jobid>` (live resource use).

## Common mistakes

| Mistake | Fix |
|---|---|
| Running compute on the login node | It has no GPU; always `srun`/`sbatch`. |
| Putting repo/venv/cache in `$HOME` | Use `/workdir2/<login>`. |
| Passing `--qos`/`--partition`/`--account` | None exist here; SLURM rejects them. Use `--gpus=N`. |
| Expecting compute nodes to reach HuggingFace | Pre-download on login; `HF_HUB_OFFLINE=1` in jobs. |
| Long-context job on node20 (24 GB) → OOM | Pin to node11/node22 (≥48 GB). |
