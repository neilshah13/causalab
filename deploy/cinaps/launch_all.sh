#!/usr/bin/env bash
# Submit every job in a manifest to the Cinaps queue — one sbatch per line.
# Run on the login node from the repo root, after bootstrap.sh.
#   HF_HOME=/workdir2/$USER/hf_cache bash deploy/cinaps/launch_all.sh [manifest.tsv]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
MANIFEST="${1:-$HERE/pilots.tsv}"

cd "$REPO_DIR"
mkdir -p slurm_logs
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$WORKDIR_BASE/$(whoami)/hf_cache}"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }
echo "+ manifest: $MANIFEST    HF_HOME: $HF_HOME"

n=0
while IFS=$'\t' read -r name gpus cmd; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    echo "+ sbatch --gpus=$gpus --job-name=cinaps_$name -- uv run $cmd"
    sbatch --gpus="$gpus" --job-name="cinaps_$name" \
           --export=ALL,HF_HOME="$HF_HOME" \
           "$HERE/job.sbatch" $cmd
    n=$((n + 1))
done < "$MANIFEST"

echo "+ submitted $n job(s). Watch: bash $HERE/monitor.sh"
