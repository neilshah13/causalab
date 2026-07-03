#!/bin/bash
# OVERNIGHT 3: completeness/extent sweep — push the sampling box further beyond the
# anchors (margin) at L31 to detect whether valid activation regions exist OUTSIDE
# the anchor hull (i.e. behaviours no prompt produced but that are still reachable).
#SBATCH --job-name=ON3_marginsweep
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
for MG in 0.6 1.0 1.5 2.5 4.0; do
  echo "==== margin=$MG ===="
  uv run python "$SDIR/code/analyses/simplex_coverage/map_subspace.py" \
    --layers "31" --k 8 --margin $MG --n-samples 30000 --tag "margin${MG}" \
    --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" --experiment-root "$ROOT"
done
echo "ON3 DONE"
