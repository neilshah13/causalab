#!/usr/bin/env bash
# Quick status of your Cinaps jobs + recent logs. Run from the repo root.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
me="$(whoami)"

echo "=== queue (squeue -u $me) ==="
squeue -u "$me" -o "%.12i %.18j %.9T %.11M %.5D %R" || true

echo "=== recent finished (sacct, today) ==="
sacct -u "$me" --format=JobID,JobName%20,State,Elapsed,MaxRSS --units=G 2>/dev/null | head -20 || true

echo "=== latest logs ($REPO_DIR/slurm_logs) ==="
ls -1t "$REPO_DIR"/slurm_logs/*.out 2>/dev/null | head -8 || echo "(none yet)"
