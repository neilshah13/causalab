#!/bin/bash
#SBATCH --job-name=N3_steercmp_fin
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
MROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
MP=$SDIR/code/methods/month_prompts/prompts.json
HP=$SDIR/code/methods/hue_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# complete the cross-layer head-to-head: months at L19+L27 (L23/31 done in S1)
uv run python "$CODE/steer_compare.py" --concept months --layers 19,27 --k 13 \
  --waypoints 20 --n-carriers 16 --n-pairs 40 --prompts "$MP" --experiment-root "$MROOT" --tag "L19_27"
# hue-wheel head-to-head (cycle restricted to populated hues)
uv run python "$CODE/steer_compare.py" --concept hues12 --layers 23,31 --k 11 \
  --waypoints 20 --n-carriers 12 --n-pairs 40 --prompts "$HP" --experiment-root "$HROOT"
echo "N3 DONE"
