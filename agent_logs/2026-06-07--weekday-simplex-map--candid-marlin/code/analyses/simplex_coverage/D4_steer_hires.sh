#!/bin/bash
#SBATCH --job-name=D4_steer_hires
#SBATCH --time=01:30:00
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
uv run python "$CODE/steer_trajectory.py" --concept weekdays --layer 31 --k 8 --carrier "A day of the week:" --steps-per-edge 20 --tag hires --prompts "$WP" --experiment-root "$WROOT"
uv run python "$CODE/steer_trajectory.py" --concept months --layer 31 --k 13 --carrier "A month of the year:" --steps-per-edge 14 --tag hires --prompts "$MP" --experiment-root "$MROOT"
uv run python "$CODE/steer_trajectory.py" --concept colors --layer 31 --k 24 --carrier "A color:" --order "Red,Orange,Yellow,Green,Blue,Purple,Red" --steps-per-edge 16 --tag hires --prompts "$CP" --experiment-root "$CROOT"
echo "D4 DONE"
