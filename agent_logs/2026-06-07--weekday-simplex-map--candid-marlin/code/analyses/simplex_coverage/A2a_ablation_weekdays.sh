#!/bin/bash
#SBATCH --job-name=A2a_abl_wk
#SBATCH --time=11:00:00
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
# continuation of A1 (12514, time-limited): remaining weekday ablations, L31, leaner samples
uv run python $CODE/recovery_test.py --concept weekdays --carrier "A day of the week:" \
    --prompts "$WP" --experiment-root "$WROOT" --layer 31 --k 8 --margin 1.5 \
    --n-samples 30000 --holdout Sun --tag "Sun_L31"
for HO in "Sat,Sun" "Mon,Tue,Wed"; do
  uv run python $CODE/recovery_test.py --concept weekdays --carrier "A day of the week:" \
      --prompts "$WP" --experiment-root "$WROOT" --layer 31 --k 8 --margin 1.5 \
      --n-samples 30000 --holdout "$HO"
done
for D in Mon Wed; do for F in 0.25 0.5 0.75; do
  uv run python $CODE/recovery_test.py --concept weekdays --carrier "A day of the week:" \
      --prompts "$WP" --experiment-root "$WROOT" --layer 31 --k 8 --margin 1.5 \
      --n-samples 15000 --holdout $D --holdout-frac $F
done; done
echo "A2a DONE"
