#!/bin/bash
# Months VALIDATION: capture (full) + a quick L31 map, to confirm the generalised
# pipeline works before committing the full overnight months run.
#SBATCH --job-name=months_val
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
ROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
uv run python "$CODE/capture_concept.py" --concept months --frame fewshot_neutral \
  --prompts "$SDIR/code/methods/month_prompts/prompts.json" --experiment-root "$ROOT"
uv run python "$CODE/map_subspace.py" --concept months --carrier "A month of the year:" \
  --layers 31 --k 10 --n-samples 4000 --tag validate \
  --prompts "$SDIR/code/methods/month_prompts/prompts.json" --experiment-root "$ROOT"
echo "MONTHS_VAL DONE"
