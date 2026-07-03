#!/bin/bash
#SBATCH --job-name=D5_sparse_gif
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
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# per-sample dump of the SPARSE chart (anchors: Mon+Thu only) — data for the recovery GIF
uv run python "$CODE/map_geometry_dump.py" --concept weekdays --layers 31 --k 8 \
  --keep Mon,Thu --margin 2.0 --n-samples 20000 --tag keepMonThu \
  --carrier "A day of the week:" --experiment-root "$WROOT"
echo "D5 DONE"
