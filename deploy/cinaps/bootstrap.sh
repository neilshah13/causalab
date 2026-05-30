#!/usr/bin/env bash
# One-time environment setup on the Cinaps LOGIN node (it has internet; compute
# nodes do not). Installs uv, syncs the venv, and pre-downloads model weights.
#
# Requires HF_TOKEN in the environment (gated Llama). Run from a clone of the
# repo on /workdir2, e.g.:
#   git clone --branch cinaps-launchers <fork> /workdir2/$USER/causalab
#   cd /workdir2/$USER/causalab && HF_TOKEN=hf_xxx bash deploy/cinaps/bootstrap.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

LOGIN="$(whoami)"
WORKDIR="$WORKDIR_BASE/$LOGIN"
REPO_DIR="$(cd "$HERE/../.." && pwd)"   # repo root (where this clone lives)
export HF_HOME="$WORKDIR/hf_cache"

[ -d "$WORKDIR" ] || { echo "ERROR: $WORKDIR missing — ask the compute team to create your workdir2." >&2; exit 1; }
mkdir -p "$HF_HOME"

# 1) uv (no system uv on Cinaps) -> ~/.local/bin
if ! command -v uv >/dev/null 2>&1; then
    echo "+ installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv install failed" >&2; exit 1; }

# 2) keep the clone on the expected branch (no-op if already there)
git -C "$REPO_DIR" fetch --quiet origin "$REPO_BRANCH" || true
git -C "$REPO_DIR" checkout --quiet "$REPO_BRANCH" 2>/dev/null || true

# 3) build the venv (uv fetches Python 3.10 itself; repo is pinned to it)
echo "+ uv sync in $REPO_DIR"
( cd "$REPO_DIR" && uv sync )

# 4) pre-stage gated weights on the login node (compute nodes are offline)
: "${HF_TOKEN:?set HF_TOKEN before bootstrap (gated Llama weights)}"
echo "+ downloading: $HF_MODELS -> $HF_HOME"
( cd "$REPO_DIR" && HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" \
    uv run python -m causalab.runner.download_models $HF_MODELS )

cat <<EOF

+ bootstrap complete
    repo:     $REPO_DIR
    HF_HOME:  $HF_HOME   (export this for launches)
    next:     cd $REPO_DIR && HF_HOME=$HF_HOME bash deploy/cinaps/launch_all.sh
EOF
