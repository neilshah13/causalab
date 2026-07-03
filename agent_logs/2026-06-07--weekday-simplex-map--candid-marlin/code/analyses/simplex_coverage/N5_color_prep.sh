#!/bin/bash
#SBATCH --job-name=N5_color_prep
#SBATCH --time=01:30:00
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
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
MP=$SDIR/code/methods/month_prompts/prompts.json
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
uv run python "$SDIR/code/methods/color_prompts/color_tokens.py" --out "$SDIR/code/methods/color_prompts/colors.json"
uv run python "$SDIR/code/methods/color_prompts/build.py" --colors "$SDIR/code/methods/color_prompts/colors.json" --out "$CP"
uv run python "$CODE/capture_concept.py" --concept colors --frame fewshot_neutral --prompts "$CP" --experiment-root "$CROOT"
uv run python "$CODE/map_subspace.py" --concept colors --carrier "A color:" --layers 31 --k 12 --n-samples 8000 --tag validate --prompts "$CP" --experiment-root "$CROOT"
echo "N5 DONE"
