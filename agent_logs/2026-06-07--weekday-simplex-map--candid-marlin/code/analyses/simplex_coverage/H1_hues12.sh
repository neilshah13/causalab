#!/bin/bash
#SBATCH --job-name=H1_hues12
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
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
HP=$SDIR/code/methods/hue_prompts/prompts.json
mkdir -p "$HROOT"
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
uv run python $SDIR/code/methods/hue_prompts/prompts.py
# 1. capture: 82 hue prompts, all layers
uv run python "$CODE/capture_concept.py" --concept hues12 --prompts "$HP" --experiment-root "$HROOT"
# 2. subspace map across layers (cross-layer per session direction)
uv run python "$CODE/map_subspace.py" --concept hues12 --prompts "$HP" --experiment-root "$HROOT" \
    --carrier "A color:" --layers 19,23,27,31 --k 11 --n-samples 20000 --margin 0.6
# 3. continuous steering around the hue wheel (cyclic default order)
for L in 23 31; do
  uv run python "$CODE/steer_trajectory.py" --concept hues12 --layer $L --k 11 --carrier "A color:" \
      --steps-per-edge 6 --tag "L${L}k11" --prompts "$HP" --experiment-root "$HROOT"
done
# 4. sparse-anchor recovery on the wheel (the test that failed on the 45-gamut)
uv run python "$CODE/sparse_recovery.py" --concept hues12 --keep Red,Green,Blue \
    --k-list 6,8,11,16 --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT"
uv run python "$CODE/sparse_recovery.py" --concept hues12 --keep Red,Yellow,Blue \
    --k-list 6,8,11,16 --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT"
# 5. leave-a-region-out on the wheel
for H in Green Blue; do
  uv run python "$CODE/recovery_test.py" --concept hues12 --holdout $H --k 11 --margin 1.5 \
      --n-samples 20000 --carrier "A color:" --prompts "$HP" --experiment-root "$HROOT" --layer 31
done
# 6. geometry dump for the ring test + offline geometry analysis
uv run python "$CODE/map_geometry_dump.py" --concept hues12 --layers 23,31 --k 11 \
    --n-samples 20000 --margin 0.6 --carrier "A color:" --experiment-root "$HROOT"
echo "H1 DONE"
