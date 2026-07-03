#!/bin/bash
#SBATCH --job-name=R1_fixes
#SBATCH --time=03:00:00
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
echo "=== N4 redo: weekday sparse (logits_to_keep fix) ==="
for KEEP in "Mon" "Sat,Sun" "Mon,Tue,Wed,Thu"; do
  uv run python "$CODE/sparse_recovery.py" --concept weekdays --layer 31 --keep "$KEEP" --k-list "3,5,6,7,8,12" --methods "cca,pca,diff" --n-samples 12000 --margin 2.0 --prompts "$WP" --experiment-root "$WROOT"
done
echo "=== months recovery at CORRECT k=13 (was false-negative at k=8) ==="
for HD in Jul Jan Apr; do
  uv run python "$CODE/recovery_test.py" --concept months --layer 31 --holdout $HD --carrier "A month of the year:" --k 13 --n-samples 25000 --margin 1.5 --tag ${HD}_k13 --prompts "$MP" --experiment-root "$MROOT"
done
echo "=== months walk (N3 truncated piece) ==="
uv run python "$CODE/perturb_map.py" --concept months --mode walk --layer 31 --n-anchors 32 --n-dirs 16 --walk-steps 60 --step-size 0.5 --prompts "$MP" --experiment-root "$MROOT"
echo "R1 DONE"
