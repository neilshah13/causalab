#!/bin/bash
# Probe both models (base + instruct) on the ~26-prompt representative subset.
# Memory precisions are REQUIRED on cinaps: with no --mem the job defaults to a
# full node's RAM (~773 GB) and sits PENDING ~1 year out, never running.
#SBATCH --job-name=wsx_probe
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
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
SCRIPT=$SDIR/code/analyses/simplex_coverage/run_coverage.py

cd "$REPO"
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# BASE: completion frames. plain = no help (baseline), fewshot_neutral = CLEAN
# one-word demo on non-day tasks (no answer leak), fewshot = REF (leaks day answers).
echo "================ PROBE llama31_8b ================"
uv run python "$SCRIPT" \
  --mode probe --model llama31_8b \
  --frames plain,fewshot_neutral,fewshot \
  --prompts "$PROMPTS" \
  --experiment-root "$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage"

# INSTRUCT: brevity instructions (CLEAN: constrain form not content) vs i_leak (REF).
echo "================ PROBE llama31_8b_instruct ================"
uv run python "$SCRIPT" \
  --mode probe --model llama31_8b_instruct \
  --frames i_oneword,i_brief,i_concise,i_short,i_word3,i_leak \
  --prompts "$PROMPTS" \
  --experiment-root "$SDIR/artifacts/weekday_simplex/llama31_8b_instruct/simplex_coverage"
echo "PROBE DONE"
