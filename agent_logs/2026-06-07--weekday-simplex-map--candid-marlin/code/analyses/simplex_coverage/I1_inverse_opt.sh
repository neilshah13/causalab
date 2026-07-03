#!/bin/bash
#SBATCH --job-name=I1_inv_opt
#SBATCH --time=04:00:00
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
MROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
MP=$SDIR/code/methods/month_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# gradient inverse: weekdays at L31 + L23 (cross-layer), months at L31
for L in 31 23; do
  uv run python "$CODE/inverse_opt.py" --concept weekdays --layer $L --k 8 \
      --steps 150 --lr 0.08 --opt-batch 64 --carrier "A day of the week:" \
      --prompts "$WP" --experiment-root "$WROOT"
done
uv run python "$CODE/inverse_opt.py" --concept months --layer 31 --k 13 \
    --steps 150 --lr 0.08 --opt-batch 64 --carrier "A month of the year:" \
    --prompts "$MP" --experiment-root "$MROOT"
echo "I1 DONE"
