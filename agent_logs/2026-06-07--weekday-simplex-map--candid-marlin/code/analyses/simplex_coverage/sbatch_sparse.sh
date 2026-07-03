#!/bin/bash
#SBATCH --job-name=sparse
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
cd /workdir2/johan.boscher/causalab
for KEEP in "Mon,Thu" "Mon,Wed,Fri,Sun" "Mon,Tue,Wed,Thu"; do
  uv run python "$SDIR/code/analyses/simplex_coverage/sparse_recovery.py" --concept weekdays --layer 31 \
    --keep "$KEEP" --k-list "4,8,12,20,40" --methods "cca,pca,diff" --n-samples 20000 --margin 2.0 \
    --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "SPARSE DONE"
