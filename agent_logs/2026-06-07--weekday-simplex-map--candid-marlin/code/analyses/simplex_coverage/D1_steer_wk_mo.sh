#!/bin/bash
#SBATCH --job-name=D1_steer_wk_mo
#SBATCH --time=03:00:00
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
for L in 19 23 27 31; do for K in 6 8; do
  uv run python "$CODE/steer_trajectory.py" --concept weekdays --layer $L --k $K --carrier "A day of the week:" --steps-per-edge 8 --tag "L${L}k${K}" --prompts "$WP" --experiment-root "$WROOT"
done; done
for L in 23 27 31; do for K in 11 13; do
  uv run python "$CODE/steer_trajectory.py" --concept months --layer $L --k $K --carrier "A month of the year:" --steps-per-edge 6 --tag "L${L}k${K}" --prompts "$MP" --experiment-root "$MROOT"
done; done
echo "D1 DONE"
