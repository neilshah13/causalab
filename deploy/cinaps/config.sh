# Cinaps launcher configuration — `source` this file.
# Secrets (HF_TOKEN) come from the environment, never from here.
# Override any value by exporting it before sourcing.

CINAPS_SSH="${CINAPS_SSH:-ssh -F $HOME/.ssh/cinaps_ssh_config cinaps}"

WORKDIR_BASE="${WORKDIR_BASE:-/workdir2}"   # NB: /workdir does not exist on Cinaps; /workdir2 does
REPO_URL="${REPO_URL:-https://github.com/neilshah13/causalab.git}"
REPO_BRANCH="${REPO_BRANCH:-cinaps-launchers}"

# Models to pre-stage on the (internet-connected) login node before any job runs.
HF_MODELS="${HF_MODELS:-meta-llama/Llama-3.1-8B}"

# Generation budget for the encoding-gate scorer (multi-token answers need > 1).
GATE_MAX_NEW_TOKENS="${GATE_MAX_NEW_TOKENS:-6}"
