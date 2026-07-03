#!/bin/bash
# Step 3: map the weekday activation subspace by perturbing anchors and patching.
# Usage: sbatch sbatch_perturb.sh <mode> <layer> [extra args...]
#   mode in {radial, walk}; layer = captured hidden-state index (29/30/31).
# Memory precisions REQUIRED on cinaps.
#SBATCH --job-name=wsx_perturb
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out

set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH
export HF_HOME=/workdir2/johan.boscher/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-radial}"
LAYER="${2:-31}"
shift 2 || true

REPO=/workdir2/johan.boscher/causalab
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
SDIR=$REPO/agent_logs/$SESSION
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd "$REPO"
echo "node=$(hostname) mode=$MODE layer=$LAYER args=$*"

uv run python "$SDIR/code/analyses/simplex_coverage/perturb_map.py" \
  --mode "$MODE" --layer "$LAYER" \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" \
  --experiment-root "$ROOT" "$@"
echo "PERTURB DONE"
