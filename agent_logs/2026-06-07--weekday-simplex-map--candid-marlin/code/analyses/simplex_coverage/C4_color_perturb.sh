#!/bin/bash
#SBATCH --job-name=C4_color_perturb
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
uv run python "$CODE/perturb_map.py" --concept colors --mode radial --layer 31 --n-anchors -1 --n-dirs 16 --radii "0,0.5,1,1.5,2,3,4,6,8" --prompts "$CP" --experiment-root "$CROOT"
uv run python "$CODE/perturb_map.py" --concept colors --mode shape --layer 31 --n-anchors -1 --n-dirs 24 --radii "0.5,1,1.5,2,3" --prompts "$CP" --experiment-root "$CROOT"
echo "C4 DONE"
