#!/bin/bash
#SBATCH --job-name=N2_geom_robust
#SBATCH --time=08:00:00
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
HROOT=$SDIR/artifacts/hues12_simplex/llama31_8b/simplex_coverage
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# geometry robustness for tomorrow's conclusions:
# (a) core-regime dumps for months + hues12 (weekday core = G2, done)
uv run python "$CODE/map_geometry_dump.py" --concept months --layers 23,31 --k 13 \
  --mode gauss --sigma-nn 0.8 --n-samples 20000 --carrier "A month of the year:" --experiment-root "$MROOT"
uv run python "$CODE/map_geometry_dump.py" --concept hues12 --layers 23,31 --k 11 \
  --mode gauss --sigma-nn 0.8 --n-samples 15000 --carrier "A color:" --experiment-root "$HROOT"
# (b) seed replicate of the weekday box dump (isometry/ring robustness)
uv run python "$CODE/map_geometry_dump.py" --concept weekdays --layers 23,31 --k 8 \
  --n-samples 20000 --margin 0.6 --seed 1 --tag seed1 \
  --carrier "A day of the week:" --experiment-root "$WROOT"
echo "N2 DONE"
