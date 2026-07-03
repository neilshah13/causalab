#!/bin/bash
#SBATCH --job-name=N3_months_perturb
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
uv run python "$CODE/perturb_map.py" --concept months --mode radial --layer 31 --n-anchors -1 --n-dirs 16 --radii "0,0.5,1,1.5,2,3,4,6,8" --prompts "$MP" --experiment-root "$MROOT"
uv run python "$CODE/perturb_map.py" --concept months --mode shape  --layer 31 --n-anchors -1 --n-dirs 24 --radii "0.5,1,1.5,2,3" --prompts "$MP" --experiment-root "$MROOT"
uv run python "$CODE/perturb_map.py" --concept months --mode walk   --layer 31 --n-anchors 32 --n-dirs 16 --walk-steps 60 --step-size 0.5 --prompts "$MP" --experiment-root "$MROOT"
echo "N3 DONE"
