#!/bin/bash
# OVERNIGHT 1: complete layer sweep of the subspace-mapping routine (weekdays),
# every 2 layers across the late stack, high sample count. Charts how the valid
# region + carrier-faithfulness evolve with depth.
#SBATCH --job-name=ON1_layersweep
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
uv run python "$SDIR/code/analyses/simplex_coverage/map_subspace.py" \
  --layers "16,18,20,22,24,26,28,30,31" --k 8 --n-samples 40000 --tag layersweep \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" --experiment-root "$ROOT"
echo "ON1 DONE"
