#!/bin/bash
# OVERNIGHT 5: MONTHS generalisation — full pipeline. Re-capture (self-contained)
# then map the behaviour<->activation subspace across late layers, a k-sweep, and a
# margin/completeness sweep, mirroring the weekday analyses. Tests that the routine
# generalises to a second (12-token) simplex.
#SBATCH --job-name=ON5_months
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
ROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
PROMPTS=$SDIR/code/methods/month_prompts/prompts.json
CAR="A month of the year:"
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
uv run python "$CODE/capture_concept.py" --concept months --frame fewshot_neutral --prompts "$PROMPTS" --experiment-root "$ROOT"
echo "==== months layer sweep ===="
uv run python "$CODE/map_subspace.py" --concept months --carrier "$CAR" --layers "16,18,20,22,24,26,28,30,31" --k 10 --n-samples 30000 --tag layersweep --prompts "$PROMPTS" --experiment-root "$ROOT"
echo "==== months k sweep (L31) ===="
for K in 3 4 6 8 10 12; do
  uv run python "$CODE/map_subspace.py" --concept months --carrier "$CAR" --layers "31" --k $K --n-samples 20000 --tag "k${K}" --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "==== months margin sweep (L31) ===="
for MG in 0.6 1.2 2.0; do
  uv run python "$CODE/map_subspace.py" --concept months --carrier "$CAR" --layers "31" --k 10 --margin $MG --n-samples 25000 --tag "margin${MG}" --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "ON5 DONE"
