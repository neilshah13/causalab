#!/bin/bash
# OVERNIGHT 2: dimensionality (k) sweep of the behaviour-relevant subspace at L25
# and L31. Finds where mapping completeness saturates (expect ~6 = |Z|-1).
#SBATCH --job-name=ON2_ksweep
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
for K in 2 3 4 5 6 8 10 12; do
  echo "==== k=$K ===="
  uv run python "$SDIR/code/analyses/simplex_coverage/map_subspace.py" \
    --layers "25,31" --k $K --n-samples 20000 --tag "k${K}" \
    --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" --experiment-root "$ROOT"
done
echo "ON2 DONE"
