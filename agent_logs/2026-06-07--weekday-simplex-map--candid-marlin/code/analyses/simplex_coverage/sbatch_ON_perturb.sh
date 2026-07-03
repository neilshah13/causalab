#!/bin/bash
# OVERNIGHT 4: extend the causal perturbation map (radial + walk) to mid layers
# (L25, L27) to complement the L29-31 maps and the L19/23/27/31 alignment story.
#SBATCH --job-name=ON4_perturb
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
PM="$SDIR/code/analyses/simplex_coverage/perturb_map.py"
ROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
PROMPTS=$SDIR/code/methods/weekday_prompts/prompts.json
RADII="0,0.25,0.5,0.75,1,1.5,2,3,4,6,8"
cd /workdir2/johan.boscher/causalab
echo "node=$(hostname)"
for L in 25 27; do
  echo "==== radial L$L ===="
  uv run python "$PM" --mode radial --layer $L --n-anchors -1 --n-dirs 16 --radii "$RADII" \
    --prompts "$PROMPTS" --experiment-root "$ROOT"
done
for L in 25 29; do
  echo "==== walk L$L ===="
  uv run python "$PM" --mode walk --layer $L --n-anchors 32 --n-dirs 16 --walk-steps 60 --step-size 0.5 \
    --prompts "$PROMPTS" --experiment-root "$ROOT"
done
for L in 25 27; do
  echo "==== shape L$L ===="
  uv run python "$PM" --mode shape --layer $L --n-anchors -1 --n-dirs 24 --radii 0.5,1,1.5,2,3 \
    --prompts "$PROMPTS" --experiment-root "$ROOT"
done
echo "ON4 DONE"
