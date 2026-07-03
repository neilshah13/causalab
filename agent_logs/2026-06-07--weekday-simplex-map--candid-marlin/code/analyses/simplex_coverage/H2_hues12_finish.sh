#!/bin/bash
#SBATCH --job-name=H2_hues12_fin
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/workdir2/johan.boscher/causalab
#SBATCH --output=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin/run/%x_%j.out
set -euo pipefail
export PATH=/home/johan.boscher/.local/bin:$PATH HF_HOME=/workdir2/johan.boscher/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SDIR=/workdir2/johan.boscher/causalab/agent_logs/2026-06-07--weekday-simplex-map--candid-marlin
CODE=$SDIR/code/analyses/simplex_coverage
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
HP=$SDIR/code/methods/hue_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# continuation of H1 (12515, time-limited): capture/map/steer done; finish sparse + recovery + geometry
uv run python "$CODE/sparse_recovery.py" --concept hues12 --keep Red,Green,Blue \
    --k-list 8,11,16 --methods pca,cca --n-samples 12000 \
    --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT"
uv run python "$CODE/sparse_recovery.py" --concept hues12 --keep Red,Yellow,Blue \
    --k-list 8,11,16 --methods pca,cca --n-samples 12000 \
    --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT"
for H in Green Blue; do
  uv run python "$CODE/recovery_test.py" --concept hues12 --holdout $H --k 11 --margin 1.5 \
      --n-samples 15000 --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT" --layer 31
done
uv run python "$CODE/map_geometry_dump.py" --concept hues12 --layers 23,31 --k 11 \
    --n-samples 20000 --margin 0.6 --carrier "A color:" --experiment-root "$HROOT"
echo "H2 DONE"
