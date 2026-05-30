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

# Restrict GPU jobs to the larger cards by excluding the 32 GB (node15) and
# 24 GB (node20) nodes — leaves node11 (96 GB) and node22 (48 GB). Long-context
# pipelines need >=48 GB; short gates fit anywhere but this matches the
# recommended nodes. Set empty to use any GPU node.
GATE_EXCLUDE_NODES="${GATE_EXCLUDE_NODES:-node15,node20}"
