#!/bin/bash
# Full step-3 map: radial perturbation on L29/L30/L31 (all anchors, finer radii)
# in one model load, then a cumulative random walk on L31. --mem REQUIRED.
#SBATCH --job-name=wsx_pall
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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
PM="$SDIR/code/analyses/simplex_coverage/perturb_map.py"
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
RADII="0,0.25,0.5,0.75,1,1.5,2,3,4,6,8"
cd "$REPO"
echo "node=$(hostname)"

for L in 29 30 31; do
  echo "================ RADIAL L$L (all anchors) ================"
  uv run python "$PM" --mode radial --layer $L --n-anchors -1 --n-dirs 16 \
    --radii "$RADII" --prompts "$PROMPTS" --experiment-root "$ROOT"
done

echo "================ WALK L31 (cumulative, reachability) ================"
uv run python "$PM" --mode walk --layer 31 --n-anchors 32 --n-dirs 16 \
  --walk-steps 60 --step-size 0.5 --prompts "$PROMPTS" --experiment-root "$ROOT"
echo "PALL DONE"
