#!/bin/bash
#SBATCH --job-name=N1b_unpr_hyb
#SBATCH --time=06:00:00
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
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
CP=$SDIR/code/methods/color_prompts/prompts.json
HP=$SDIR/code/methods/hue_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# HYBRID steer-to-unprompted: box exploration finds the basin, gradient sharpens it.
# (N1 showed gradient-from-linear-init stalls: weekdays 3/5 vs box-sampling's 5/5.)
for L in 31 27; do
  uv run python "$CODE/steer_unprompted.py" --concept colors --keep Red,Green,Blue \
      --layer $L --k 12 --method pca --init both --carrier "A color:" --prompts "$CP" \
      --experiment-root "$CROOT" --tag "RGB_L${L}pca_hybrid"
done
uv run python "$CODE/steer_unprompted.py" --concept colors \
    --keep Red,Orange,Yellow,Green,Blue,Purple,Pink,Brown \
    --layer 31 --k 12 --method pca --init both --carrier "A color:" --prompts "$CP" \
    --experiment-root "$CROOT" --tag "keep8_L31pca_hybrid"
uv run python "$CODE/steer_unprompted.py" --concept hues12 --keep Red,Green,Blue \
    --layer 31 --k 8 --method pca --init both --carrier "A color:" --prompts "$HP" \
    --experiment-root "$HROOT" --tag "RGB_L31pca_hybrid"
uv run python "$CODE/steer_unprompted.py" --concept weekdays --keep Mon,Thu \
    --layer 31 --k 8 --method pca --init both --carrier "A day of the week:" --prompts "$WP" \
    --experiment-root "$WROOT" --tag "MonThu_L31pca_hybrid"
echo "N1b DONE"
