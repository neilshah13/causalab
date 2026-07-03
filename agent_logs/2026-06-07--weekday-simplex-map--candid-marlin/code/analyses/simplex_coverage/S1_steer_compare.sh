#!/bin/bash
#SBATCH --job-name=S1_steer_cmp
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
WROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
MROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
MP=$SDIR/code/methods/month_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
uv run python "$CODE/steer_compare.py" --concept weekdays --layers 19,23,27,31 --k 8 \
  --waypoints 20 --n-carriers 16 --prompts "$WP" --experiment-root "$WROOT"
uv run python "$CODE/steer_compare.py" --concept months --layers 23,31 --k 13 \
  --waypoints 20 --n-carriers 16 --n-pairs 40 --prompts "$MP" --experiment-root "$MROOT"
echo "S1 DONE"
