#!/usr/bin/env bash
# Generic Cinaps job step. Submitted by launch_all.sh as:
#   sbatch --gpus=N --job-name=cinaps_<name> job.sbatch <command...>
# The <command...> runs as `uv run <command...>` from the repo root.
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workdir2/$(whoami)/hf_cache}"
export HF_HUB_OFFLINE=1          # compute nodes have no internet; weights are pre-staged

echo "+ node=$(hostname) job=${SLURM_JOB_NAME:-?} jobid=${SLURM_JOB_ID:-?}"
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"

uv run "$@"
