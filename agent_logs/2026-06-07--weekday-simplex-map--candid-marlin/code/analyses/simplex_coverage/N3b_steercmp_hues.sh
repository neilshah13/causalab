#!/bin/bash
#SBATCH --job-name=N3b_cmp_hues
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
HP=$SDIR/code/methods/hue_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# N3 retry: hues12 head-to-head (pairs now in populated-cycle index space)
uv run python "$CODE/steer_compare.py" --concept hues12 --layers 23,31 --k 11 \
  --waypoints 20 --n-carriers 12 --n-pairs 40 --prompts "$HP" --experiment-root "$HROOT"
echo "N3b DONE"
