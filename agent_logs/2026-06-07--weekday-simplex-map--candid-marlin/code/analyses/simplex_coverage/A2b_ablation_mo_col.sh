#!/bin/bash
#SBATCH --job-name=A2b_abl_mocol
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
MROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
MP=$SDIR/code/methods/month_prompts/prompts.json
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"
# continuation of A1 (12514, time-limited): months remaining regions at k=13, leaner samples
for MO in Feb Mar May Jun Aug Sep Oct Nov Dec; do
  uv run python $CODE/recovery_test.py --concept months --carrier "A month of the year:" \
      --prompts "$MP" --experiment-root "$MROOT" --layer 31 --k 13 --margin 1.5 \
      --n-samples 15000 --holdout $MO --tag "${MO}_k13"
done
# colour family-exclusion deletions (no synonym leakage)
uv run python $CODE/recovery_test.py --concept colors --carrier "A color:" \
    --prompts "$CP" --experiment-root "$CROOT" --layer 31 --k 24 --margin 1.5 \
    --n-samples 12000 --holdout "Blue,Navy,Cyan,Azure,Teal,Turquoise,Sapphire,Aqua" --tag "BlueFam_k24"
uv run python $CODE/recovery_test.py --concept colors --carrier "A color:" \
    --prompts "$CP" --experiment-root "$CROOT" --layer 31 --k 24 --margin 1.5 \
    --n-samples 12000 --holdout "Red,Crimson,Scarlet,Ruby,Rose,Salmon,Coral" --tag "RedFam_k24"
echo "A2b DONE"
