#!/bin/bash
#SBATCH --job-name=C2_color_sparse
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
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
for KEEP in "Red,Blue,Green,Yellow" "Red,Orange,Yellow,Green,Blue,Purple"; do
  uv run python "$CODE/sparse_recovery.py" --concept colors --layer 31 --carrier "A color:" --keep "$KEEP" --k-list "8,16,24,32,40" --methods "cca,pca,diff" --n-samples 15000 --margin 2.0 --prompts "$CP" --experiment-root "$CROOT"
done
echo "C2 DONE"
