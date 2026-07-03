#!/bin/bash
#SBATCH --job-name=G1_geom_dump
#SBATCH --time=02:30:00
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
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
uv run python "$CODE/map_geometry_dump.py" --concept weekdays --layers 19,23,27,31 --k 8 \
  --n-samples 30000 --margin 0.6 --carrier "A day of the week:" --experiment-root "$WROOT"
uv run python "$CODE/map_geometry_dump.py" --concept months --layers 23,31 --k 13 \
  --n-samples 30000 --margin 0.6 --carrier "A month of the year:" --experiment-root "$MROOT"
echo "G1 DONE"
