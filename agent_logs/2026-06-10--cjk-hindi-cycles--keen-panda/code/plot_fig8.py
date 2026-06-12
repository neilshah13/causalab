#!/usr/bin/env python3
"""Standalone Fig 8: cross-lingual steering coherence vs same-language baseline.

Data is hardcoded from completed Cinaps jobs 12573-12575 (2026-06-12).
Run with:
    uv run python agent_logs/2026-06-10--cjk-hindi-cycles--keen-panda/code/plot_fig8.py
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_DIR = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(SESSION_DIR, "result", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Cross-lingual steering coherence (jobs 12573–12575)
cross_lingual = {
    ("en", "zh"): {"mean": 0.9631, "se": 0.0005},
    ("zh", "ja"): {"mean": 0.7701, "se": 0.0235},
    ("en", "fr"): {"mean": 0.9802, "se": 0.0004},
}

# Same-language geometric coherence baselines (from path_steering)
same_lang = {
    "en": {"mean": 0.7829, "se": 0.0033},
    "zh": {"mean": 0.9664, "se": 0.0005},
    "ja": {"mean": 0.9785, "se": 0.0005},
    "fr": {"mean": 0.7445, "se": 0.0003},
}

# Subspace overlap (principal-angle metric)
overlap = {("en", "zh"): 0.242, ("zh", "ja"): 0.350, ("en", "fr"): 0.023}

target_langs = ["zh", "ja", "fr"]
target_labels = {"zh": "Chinese\n(ZH target)", "ja": "Japanese\n(JA target)", "fr": "French\n(FR target)"}
source_labels = {("en", "zh"): "EN→ZH", ("zh", "ja"): "ZH→JA", ("en", "fr"): "EN→FR"}
cross_keys = [("en", "zh"), ("zh", "ja"), ("en", "fr")]

fig, ax = plt.subplots(figsize=(7, 4.8))
x = np.arange(len(target_langs))
width = 0.36

same_means = [same_lang[t]["mean"] for t in target_langs]
same_ses   = [same_lang[t]["se"]   for t in target_langs]
cross_means = [cross_lingual[k]["mean"] for k in cross_keys]
cross_ses   = [cross_lingual[k]["se"]   for k in cross_keys]

bars_same = ax.bar(
    x - width / 2, same_means, width,
    yerr=same_ses, capsize=4,
    color="#90CAF9", edgecolor="#1565C0", linewidth=1.2,
    label="Same-language baseline",
)
bars_cross = ax.bar(
    x + width / 2, cross_means, width,
    yerr=cross_ses, capsize=4,
    color="#FFCC80", edgecolor="#E65100", linewidth=1.2,
    label="Cross-lingual steering",
)

for i, (k, bar) in enumerate(zip(cross_keys, bars_cross)):
    ov = overlap[k]
    lbl = source_labels[k]
    top = bar.get_height() + cross_ses[i] + 0.015
    ax.text(
        bar.get_x() + bar.get_width() / 2, top,
        f"{lbl}\noverlap={ov:.3f}",
        ha="center", va="bottom", fontsize=7.5,
        color="#E65100", fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels([target_labels[t] for t in target_langs], fontsize=11)
ax.set_ylabel("Geometric steering coherence", fontsize=11)
ax.set_ylim(0, 1.18)
ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.4)
ax.legend(fontsize=10, loc="upper left")
ax.set_title(
    "Cross-lingual steering coherence vs same-language baseline\n"
    "(Llama 3.1 8B · weekday arithmetic · Layer 28 last-token · geometric paths)",
    fontsize=10,
)

ax.text(
    0.98, 0.04,
    "EN ring (isometry r=0.989) steers FR better\n"
    "than FR's own spline (r=0.087) — even with\n"
    "only 2.3% subspace overlap.",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    color="#555555", style="italic",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9),
)

fig.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES_DIR, f"fig8_steering_coherence.{ext}")
    kw = {"bbox_inches": "tight"}
    if ext == "png":
        kw["dpi"] = 150
    fig.savefig(path, **kw)
    print(f"Saved {path}")

plt.close(fig)
print("Done.")
