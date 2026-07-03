#!/bin/bash
# Leave-a-region-out recovery test: delete each of several behavioural regions, rebuild
# the map from the rest, measure recovery (+ matched random control). One model load.
#SBATCH --job-name=recovery
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
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
for D in Wed Sat Mon Fri; do
  echo "==== holdout $D ===="
  uv run python "$CODE/recovery_test.py" --concept weekdays --layer 31 --holdout $D \
    --k 8 --n-samples 30000 --margin 1.5 --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "RECOVERY DONE"
