#!/bin/bash
# Robust subspace-mapping routine. Usage: sbatch sbatch_mapsub.sh "<layers>" <n_samples>
#   e.g. sbatch sbatch_mapsub.sh "31" 8000   (validation)
#        sbatch sbatch_mapsub.sh "19,23,27,31" 20000   (layer sweep)
# --mem REQUIRED on cinaps.
#SBATCH --job-name=wsx_mapsub
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out

set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH
export HF_HOME=/workdir2/johan.boscher/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LAYERS="${1:-31}"
NS="${2:-20000}"
REPO=/workdir2/johan.boscher/causalab
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
SDIR=$REPO/agent_logs/$SESSION
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd "$REPO"
echo "node=$(hostname) layers=$LAYERS n_samples=$NS"

uv run python "$SDIR/code/analyses/simplex_coverage/map_subspace.py" \
  --layers "$LAYERS" --n-samples "$NS" \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" \
  --experiment-root "$ROOT"
echo "MAPSUB DONE"
