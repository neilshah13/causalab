#!/bin/bash
#SBATCH --job-name=D2_steer_colors
#SBATCH --time=02:30:00
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
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
MP=$SDIR/code/methods/month_prompts/prompts.json
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
for L in 27 31; do for K in 16 24 32; do
  uv run python "$CODE/steer_trajectory.py" --concept colors --layer $L --k $K --carrier "A color:" --order "Red,Orange,Yellow,Green,Blue,Purple,Red" --steps-per-edge 8 --tag "rainbow_L${L}k${K}" --prompts "$CP" --experiment-root "$CROOT"
done; done
uv run python "$CODE/steer_trajectory.py" --concept colors --layer 31 --k 24 --carrier "A color:" --order "Red,Blue" --steps-per-edge 24 --tag "RedBlue" --prompts "$CP" --experiment-root "$CROOT"
echo "D2 DONE"
