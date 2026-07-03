#!/bin/bash
# OVERNIGHT 6: highest-resolution "most complete" weekday maps at L29 & L31
# (120k samples, k=6, margin 1.0), plus two reseeded L31 maps to check map
# stability/robustness.
#SBATCH --job-name=ON6_complete
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
uv run python "$CODE/map_subspace.py" --layers "29,31" --k 6 --n-samples 120000 --margin 1.0 --tag complete --prompts "$PROMPTS" --experiment-root "$ROOT"
for S in 1 2; do
  uv run python "$CODE/map_subspace.py" --layers "31" --k 8 --n-samples 20000 --seed $S --tag "seed${S}" --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "ON6 DONE"
