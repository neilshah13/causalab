#!/bin/bash
# Full coverage run. Runs BOTH chosen combos in one job (sequential, ~2 model
# loads): the base model with a clean neutral few-shot is the real deliverable
# (preserves belief spread); the instruct model with a one-word instruction is a
# comparison that documents the vertex-collapse. Memory precisions REQUIRED
# (no --mem -> job defaults to a full node's RAM and never schedules).
#SBATCH --job-name=wsx_full
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
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
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
SCRIPT=$SDIR/code/analyses/simplex_coverage/run_coverage.py

cd "$REPO"
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 1) BASE + clean neutral few-shot  (the deliverable: spread-preserving)
echo "================ FULL llama31_8b / fewshot_neutral ================"
uv run python "$SCRIPT" \
  --mode full --model llama31_8b --frame fewshot_neutral \
  --prompts "$PROMPTS" \
  --experiment-root "$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage"

# 2) INSTRUCT + one-word brevity  (comparison: vertex-collapse)
echo "================ FULL llama31_8b_instruct / i_oneword ================"
uv run python "$SCRIPT" \
  --mode full --model llama31_8b_instruct --frame i_oneword \
  --prompts "$PROMPTS" \
  --experiment-root "$SDIR/artifacts/weekday_simplex/llama31_8b_instruct/simplex_coverage"
echo "FULL DONE"
