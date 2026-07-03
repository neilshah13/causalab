#!/bin/bash
#SBATCH --job-name=D6_gen_demo
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
WROOT=$SDIR/artifacts/weekday_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
for L in 23 27; do
  uv run python "$CODE/steer_gen_demo.py" --layer $L --k 8 --targets Wed,Sat,Sun \
      --prompts "$WP" --experiment-root "$WROOT"
done
echo "D6 DONE"
