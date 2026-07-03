#!/bin/bash
# Step 2: capture last-token residual activations (base, winning frame) for all
# prompts, then run the behaviour-vs-activation SUBSPACE comparison (linear; no
# manifolds). Memory precisions REQUIRED on cinaps.
#SBATCH --job-name=wsx_acts
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
CODE=$SDIR/code/analyses/simplex_coverage
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
cd "$REPO"
echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "================ CAPTURE activations (base, fewshot_neutral) ================"
uv run python "$CODE/capture_activations.py" \
  --model llama31_8b --frame fewshot_neutral \
  --prompts "$SDIR/code/methods/weekday_prompts/prompts.json" \
  --experiment-root "$ROOT"

echo "================ SUBSPACE comparison (behaviour vs activation) ================"
uv run python "$CODE/subspace_compare.py" --experiment-root "$ROOT"
echo "ACTS DONE"
