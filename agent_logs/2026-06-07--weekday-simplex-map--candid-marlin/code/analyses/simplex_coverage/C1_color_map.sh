#!/bin/bash
#SBATCH --job-name=C1_color_map
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
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
uv run python "$CODE/map_subspace.py" --concept colors --carrier "A color:" --layers "19,23,27,31" --k 16 --n-samples 20000 --tag layersweep --prompts "$CP" --experiment-root "$CROOT"
for K in 8 16 24 32; do uv run python "$CODE/map_subspace.py" --concept colors --carrier "A color:" --layers 31 --k $K --n-samples 15000 --tag "k${K}" --prompts "$CP" --experiment-root "$CROOT"; done
echo "C1 DONE"
