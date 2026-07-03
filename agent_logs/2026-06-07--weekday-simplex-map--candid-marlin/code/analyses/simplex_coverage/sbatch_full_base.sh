#!/bin/bash
# Base-only full run (fast iteration; instruct result is unaffected by the
# base few-shot). Memory precisions REQUIRED on cinaps.
#SBATCH --job-name=wsx_base
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out

set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH
export HF_HOME=/workdir2/johan.boscher/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/workdir2/johan.boscher/causalab
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
SDIR=$REPO/agent_logs/$SESSION
cd "$REPO"
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python "$SDIR/code/analyses/simplex_coverage/run_coverage.py" \
  --mode full --model llama31_8b --frame fewshot_neutral \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" \
  --experiment-root "$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage"
echo "BASE FULL DONE"
