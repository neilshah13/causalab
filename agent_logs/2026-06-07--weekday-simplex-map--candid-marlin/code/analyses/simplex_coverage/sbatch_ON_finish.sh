#!/bin/bash
# Finish the two truncated overnight jobs: ON3 remaining weekday margins (1.5,2.5,4.0)
# and ON4 remaining perturb runs. Pinned to the big/fast GPUs (>=48GB) to avoid the
# slow node20, with a generous time budget.
#SBATCH --job-name=ON_finish
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --exclude=node20
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
echo "=== ON3 remaining margins ==="
for MG in 1.5 2.5 4.0; do
  uv run python "$CODE/map_subspace.py" --layers "31" --k 8 --margin $MG --n-samples 30000 --tag "margin${MG}" --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "=== ON4 remaining perturb ==="
uv run python "$CODE/perturb_map.py" --mode radial --layer 27 --n-anchors -1 --n-dirs 12 --radii "0,0.5,1,1.5,2,3,4,6,8" --prompts "$PROMPTS" --experiment-root "$ROOT"
uv run python "$CODE/perturb_map.py" --mode walk --layer 25 --n-anchors 32 --n-dirs 16 --walk-steps 60 --step-size 0.5 --prompts "$PROMPTS" --experiment-root "$ROOT"
uv run python "$CODE/perturb_map.py" --mode walk --layer 29 --n-anchors 32 --n-dirs 16 --walk-steps 60 --step-size 0.5 --prompts "$PROMPTS" --experiment-root "$ROOT"
uv run python "$CODE/perturb_map.py" --mode shape --layer 27 --n-anchors -1 --n-dirs 24 --radii 0.5,1,1.5,2,3 --prompts "$PROMPTS" --experiment-root "$ROOT"
echo "ON_FINISH DONE"
