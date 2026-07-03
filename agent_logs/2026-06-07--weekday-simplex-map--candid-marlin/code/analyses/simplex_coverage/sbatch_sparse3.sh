#!/bin/bash
#SBATCH --job-name=sparse3
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd /workdir2/johan.boscher/causalab
# the hard contiguous-extrapolation case: anchor Mon-Thu, recover Fri/Sat/Sun (all on one side)
uv run python "$SDIR/code/analyses/simplex_coverage/sparse_recovery.py" --concept weekdays --layer 31 \
  --keep "Mon,Tue,Wed,Thu" --k-list "4,8,12,20" --methods "cca,pca,diff" --n-samples 15000 --margin 2.0 \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" --experiment-root "$ROOT"
echo "SPARSE3 DONE"
