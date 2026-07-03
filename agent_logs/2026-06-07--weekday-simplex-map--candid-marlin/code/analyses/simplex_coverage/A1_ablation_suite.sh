#!/bin/bash
#SBATCH --job-name=A1_ablation
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
MROOT=$SDIR/artifacts/months_simplex/llama31_8b/simplex_coverage
CROOT=$SDIR/artifacts/colors_simplex/llama31_8b/simplex_coverage
WP=$SDIR/code/methods/weekday_prompts/prompts.json
MP=$SDIR/code/methods/month_prompts/prompts.json
CP=$SDIR/code/methods/color_prompts/prompts.json
cd /workdir2/johan.boscher/causalab; echo "node=$(hostname)"

WK="uv run python $CODE/recovery_test.py --concept weekdays --carrier"
# --- weekdays: remaining single regions, L31 + cross-layer L23 ---
for D in Tue Thu Sun; do for L in 31 23; do
  $WK "A day of the week:" --prompts "$WP" --experiment-root "$WROOT" \
      --layer $L --k 8 --margin 1.5 --n-samples 30000 --holdout $D --tag "${D}_L${L}"
done; done
# --- weekdays: multi-region deletions (weekend; contiguous arc) ---
for HO in "Sat,Sun" "Mon,Tue,Wed"; do
  $WK "A day of the week:" --prompts "$WP" --experiment-root "$WROOT" \
      --layer 31 --k 8 --margin 1.5 --n-samples 30000 --holdout "$HO"
done
# --- weekdays: titration (fraction of the region deleted) on Mon (dominant) + Wed ---
for D in Mon Wed; do for F in 0.25 0.5 0.75; do
  $WK "A day of the week:" --prompts "$WP" --experiment-root "$WROOT" \
      --layer 31 --k 8 --margin 1.5 --n-samples 30000 --holdout $D --holdout-frac $F
done; done
# --- months: the 9 regions never tested, at the corrected k=13 ---
for MO in Feb Mar May Jun Aug Sep Oct Nov Dec; do
  uv run python $CODE/recovery_test.py --concept months --carrier "A month of the year:" \
      --prompts "$MP" --experiment-root "$MROOT" \
      --layer 31 --k 13 --margin 1.5 --n-samples 25000 --holdout $MO --tag "${MO}_k13"
done
# --- colours: family-exclusion deletions (the honest test — no synonym leakage) ---
uv run python $CODE/recovery_test.py --concept colors --carrier "A color:" \
    --prompts "$CP" --experiment-root "$CROOT" --layer 31 --k 24 --margin 1.5 \
    --n-samples 20000 --holdout "Blue,Navy,Cyan,Azure,Teal,Turquoise,Sapphire,Aqua" --tag "BlueFam_k24"
uv run python $CODE/recovery_test.py --concept colors --carrier "A color:" \
    --prompts "$CP" --experiment-root "$CROOT" --layer 31 --k 24 --margin 1.5 \
    --n-samples 20000 --holdout "Red,Crimson,Scarlet,Ruby,Rose,Salmon,Coral" --tag "RedFam_k24"
echo "A1 DONE"
