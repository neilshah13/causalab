#!/bin/bash
#SBATCH --job-name=G2_geom_core
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
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# core-regime dump: Gaussian clouds around anchors (0-2 NN), completes the chart-breakdown curve
uv run python "$CODE/map_geometry_dump.py" --concept weekdays --layers 23,31 --k 8 \
  --mode gauss --sigma-nn 0.8 --n-samples 20000 --carrier "A day of the week:" --experiment-root "$WROOT"
echo "G2 DONE"
