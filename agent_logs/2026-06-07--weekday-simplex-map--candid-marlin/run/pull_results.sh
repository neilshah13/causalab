#!/bin/bash
# One-command pull: bring ALL cinaps results local + render every figure PDF to PNG
# for offline viewing. Run from the repo root on the local machine.
set -euo pipefail
SESSION=2026-06-07--weekday-simplex-map--candid-marlin
SSHCFG="$HOME/cinaps_ssh_config"
REMOTE=cinaps:/workdir2/johan.boscher/causalab/agent_logs/$SESSION

echo "[1/3] rsync artifacts + run logs + colour json from cinaps ..."
rsync -az -e "ssh -F $SSHCFG" "$REMOTE/artifacts" "agent_logs/$SESSION/"
rsync -az -e "ssh -F $SSHCFG" "$REMOTE/run"       "agent_logs/$SESSION/"
rsync -az -e "ssh -F $SSHCFG" "$REMOTE/code/methods/color_prompts/" "agent_logs/$SESSION/code/methods/color_prompts/"

echo "[2/3] render figure PDFs -> PNG (offline viewing) ..."
python3 - <<'PY'
import os, glob, fitz
S="2026-06-07--weekday-simplex-map--candid-marlin"; base=f"agent_logs/{S}"
pdfs=glob.glob(f"{base}/artifacts/**/figures/*.pdf",recursive=True)+glob.glob(f"{base}/result/figures/*.pdf")
n=0
for p in pdfs:
    png=p[:-4]+".png"
    if os.path.exists(png) and os.path.getmtime(png)>=os.path.getmtime(p): continue
    try: fitz.open(p)[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(png); n+=1
    except Exception: pass
print(f"  rendered {n} new PNGs ({len(pdfs)} PDFs total)")
PY

echo "[3/3] done. Browse:"
echo "  result/REPORT.md            (Steps 1-8 narrative)"
echo "  run/DAY3_JOBS.md            (today's job table + how to read)"
echo "  artifacts/{weekday,months,colors}_simplex/llama31_8b/simplex_coverage/figures/*.png"
