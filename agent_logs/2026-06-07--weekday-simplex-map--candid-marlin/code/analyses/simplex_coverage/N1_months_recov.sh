#!/bin/bash
#SBATCH --job-name=N1_months_recov
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
for HD in Jul Jan Apr; do
  uv run python "$CODE/recovery_test.py" --concept months --layer 31 --holdout $HD --carrier "A month of the year:" --k 8 --n-samples 25000 --margin 1.5 --prompts "$MP" --experiment-root "$MROOT"
done
for L in 31 29; do
  uv run python "$CODE/inverse_map.py" --concept months --layer $L --carrier "A month of the year:" --k 10 --prompts "$MP" --experiment-root "$MROOT"
done
echo "N1 DONE"
