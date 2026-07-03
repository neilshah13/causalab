#!/bin/bash
#SBATCH --job-name=C3_color_recov_inv
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
for HD in Blue Red Green; do uv run python "$CODE/recovery_test.py" --concept colors --layer 31 --holdout $HD --carrier "A color:" --k 16 --n-samples 25000 --margin 1.5 --prompts "$CP" --experiment-root "$CROOT"; done
uv run python "$CODE/inverse_map.py" --concept colors --layer 31 --carrier "A color:" --k 16 --prompts "$CP" --experiment-root "$CROOT"
echo "C3 DONE"
