#!/usr/bin/env python3
"""Multilingual manifold visualizations for the CJK/Hindi cycles session.

Restyled to match the talk-deck figure spec (clean, presentation-grade,
DejaVu Sans, semantic palette, twilight weekday coloring). Produces eight
figures (PNG + PDF) under FIGURES_DIR. Run with:

    uv run python agent_logs/2026-06-10--cjk-hindi-cycles--keen-panda/code/plot_manifolds.py

Geometry for fig1/fig2/fig3 is read from ``recovered_manifold_data.json``:
the exact plotted marker positions recovered from the original vector PDFs
(the source ``manifold_data.json`` was lost). PCA axis units are arbitrary,
so the recovered positions are a faithful record of the manifold shapes.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from scipy.interpolate import splprep, splev

# ── Paths ──────────────────────────────────────────────────────────────────────
SESSION_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(SESSION_DIR, "code", "recovered_manifold_data.json")
FIGURES_DIR = os.path.join(SESSION_DIR, "result", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(DATA_PATH) as f:
    REC = json.load(f)

# ── Deck style spec ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
})

# Semantic palette — by MEANING, not by slot.
TEAL  = "#2A9D8F"   # good / faithful / geodesic
CORAL = "#E76F51"   # our method / the highlighted finding (eye lands here)
GRAY  = "#B0BEC5"   # weak / linear / recessive baseline
CHAR  = "#264653"   # the paper / secondary method (dark)
GOLD  = "#E9C46A"   # spare accent

# Consistent per-language palette (used in every figure):
#   EN = CORAL  — the headline closed ring (eye lands here)
#   ZH = TEAL   — functionally faithful (high coherence)
#   JA = CHAR   — secondary (dark)
#   FR = GRAY   — weakest / non-isometric / recessive
LANGS = ["en", "fr", "zh", "ja"]
LANG_LABELS = {"en": "English", "fr": "French", "zh": "Chinese", "ja": "Japanese"}
LANG_COLORS = {"en": CORAL, "fr": GRAY, "zh": TEAL, "ja": CHAR}
LANG_RING = REC["ring"]

# Cyclic weekday colors → twilight (NOT hsv).
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYCOL = [plt.cm.twilight(i / 7.0) for i in range(7)]

ISOMETRY_R = {"en": 0.989, "fr": 0.087, "zh": 0.317, "ja": -0.101}
COHERENCE  = {"en": 0.783, "fr": 0.745, "zh": 0.966, "ja": 0.979}
GATE_ACC   = {"en": 93.9, "fr": 79.6, "zh": 67.3, "ja": 63.3}
PULLBACK   = {"en": (0.687, 0.419), "fr": (0.266, 0.530)}  # (geodesic, linear)
OVERLAP    = REC["overlap"]


# ── Helpers ─────────────────────────────────────────────────────────────────────
def smooth_curve(pts: np.ndarray, closed: bool, n: int = 300) -> np.ndarray:
    if closed:
        p = np.vstack([pts, pts[0]])
        tck, _ = splprep([p[:, 0], p[:, 1]], s=0, per=True)
    else:
        k = min(3, len(pts) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=False, k=k)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def order_for(pts: np.ndarray, is_ring: bool) -> np.ndarray:
    """Ring → sort by angle (closed loop); spline → weekday order (as stored)."""
    if is_ring:
        c = pts.mean(0)
        return np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return np.arange(len(pts))


def draw_manifold(ax, pts, color, is_ring, marker_colors=None, ms=140, lw=2.4):
    """Connecting curve + white-edged circle markers for one manifold."""
    order = order_for(pts, is_ring)
    curve = smooth_curve(pts[order], closed=is_ring)
    ax.plot(curve[:, 0], curve[:, 1], color=color, lw=lw,
            ls="-" if is_ring else (0, (5, 3)), alpha=0.85, zorder=2)
    mc = marker_colors if marker_colors is not None else color
    ax.scatter(pts[:, 0], pts[:, 1], s=ms, c=mc, edgecolors="white",
               linewidths=1.5, zorder=3)


def save(fig, name, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), facecolor="white")
    print(f"Saved {name}.png / .pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Per-language manifold panels
# ══════════════════════════════════════════════════════════════════════════════
def fig1_language_manifolds():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    for ax, lang in zip(axes.flat, LANGS):
        pts = np.array(REC["fig1"][lang])
        is_ring = LANG_RING[lang]
        draw_manifold(ax, pts, LANG_COLORS[lang], is_ring,
                      marker_colors=DAYCOL, ms=150)
        shape = "ring" if is_ring else "open"
        ax.set_title(f"{LANG_LABELS[lang]} — {shape} (r={ISOMETRY_R[lang]:+.2f})",
                     color=LANG_COLORS[lang], pad=6)
        ax.set_aspect("equal")
        ax.axis("off")

    # Reserve a clear strip at the bottom for the shared weekday colorbar.
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.13,
                        wspace=0.05, hspace=0.16)
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.018])
    sm = ScalarMappable(cmap=plt.cm.twilight, norm=plt.Normalize(-0.5, 6.5))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(range(7))
    cbar.set_ticklabels(WEEKDAYS)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8)
    fig.suptitle("Weekday manifolds by language", fontsize=11, fontweight="bold")
    save(fig, "fig1_language_manifolds", tight=False)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Joint embedding (all four languages)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_joint_embedding():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for lang in LANGS:
        pts = np.array(REC["fig2"][lang])
        is_ring = LANG_RING[lang]
        draw_manifold(ax, pts, LANG_COLORS[lang], is_ring, ms=95, lw=2.2)
        c = pts.mean(0)
        ax.annotate(LANG_LABELS[lang], (c[0], c[1]), xytext=(0, 18),
                    textcoords="offset points", color=LANG_COLORS[lang],
                    fontsize=10, fontweight="bold", ha="center", va="bottom")

    handles = [
        mlines.Line2D([], [], color=LANG_COLORS[l], lw=2.4,
                      ls="-" if LANG_RING[l] else (0, (5, 3)),
                      marker="o", mec="white", mew=1.2, ms=8,
                      label=f"{LANG_LABELS[l]} ({'ring' if LANG_RING[l] else 'open'})")
        for l in LANGS
    ]
    ax.legend(handles=handles, loc="lower left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("joint PC 1"); ax.set_ylabel("joint PC 2")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title("Joint weekday embedding", pad=6)
    save(fig, "fig2_joint_embedding")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Pairwise subspace alignment
# ══════════════════════════════════════════════════════════════════════════════
def fig3_subspace_alignment():
    pairs = [("en", "zh"), ("en", "fr"), ("zh", "ja")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2))
    for ax, (la, lb) in zip(axes, pairs):
        key = f"{la}_{lb}"
        for lang in (la, lb):
            pts = np.array(REC["fig3"][key][lang])
            draw_manifold(ax, pts, LANG_COLORS[lang], LANG_RING[lang], ms=85, lw=2.2)
            c = pts.mean(0)
            ax.annotate(LANG_LABELS[lang], (c[0], c[1]), xytext=(0, 16),
                        textcoords="offset points", color=LANG_COLORS[lang],
                        fontsize=9, fontweight="bold", ha="center", va="bottom")
        ov = OVERLAP[la][lb]
        ax.set_title(f"{LANG_LABELS[la]} ↔ {LANG_LABELS[lb]}", pad=6)
        ax.text(0.5, 0.02, f"subspace overlap = {ov:.3f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#555555")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("pair PC 1"); ax.set_ylabel("pair PC 2")
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
    fig.suptitle("Pairwise subspace alignment", fontsize=11, fontweight="bold")
    save(fig, "fig3_subspace_alignment")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Subspace overlap heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig4_overlap_heatmap():
    n = len(LANGS)
    M = np.array([[OVERLAP[a][b] for b in LANGS] for a in LANGS])
    cmap = LinearSegmentedColormap.from_list("teal_seq", ["#FFFFFF", TEAL, CHAR])

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    labels = [LANG_LABELS[l] for l in LANGS]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if v > 0.5 else "#264653")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    ax.set_title("Weekday-ring subspace overlap", pad=6)
    save(fig, "fig4_overlap_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Pullback: geodesic vs linear
# ══════════════════════════════════════════════════════════════════════════════
def fig5_pullback_comparison():
    langs = ["en", "fr"]
    cats = [LANG_LABELS[l] for l in langs]
    geo = [PULLBACK[l][0] for l in langs]
    lin = [PULLBACK[l][1] for l in langs]
    x = np.arange(len(cats)); w = 0.38

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - w/2, geo, w*0.9, label="Geodesic", color=TEAL, edgecolor="white", lw=0.5)
    ax.bar(x + w/2, lin, w*0.9, label="Linear", color=GRAY, edgecolor="white", lw=0.5)
    for j, v in enumerate(geo):
        ax.text(j - w/2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("path recapitulation r² (↑ better)")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Geodesic vs linear pullback", pad=6)
    save(fig, "fig5_pullback_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — English safety direction transfer
# ══════════════════════════════════════════════════════════════════════════════
def fig6_safety_projection():
    order = ["en", "ja", "zh", "fr"]   # descending overlap
    vals = [OVERLAP["en"][l] * 100 for l in order]
    labels = [LANG_LABELS[l] + ("\n(source)" if l == "en" else "") for l in order]
    colors = [LANG_COLORS[l] for l in order]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    y = np.arange(len(order))[::-1]
    bars = ax.barh(y, vals, color=colors, edgecolor="white", lw=0.5, height=0.72)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("retained norm (%) of English safety direction")
    ax.set_title("English safety direction transfer", pad=6)
    save(fig, "fig6_safety_projection")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Isometry vs steering coherence
# ══════════════════════════════════════════════════════════════════════════════
def fig7_isometry_coherence():
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for lang in LANGS:
        iso, coh = ISOMETRY_R[lang], COHERENCE[lang]
        ax.scatter(iso, coh, s=220, c=LANG_COLORS[lang],
                   edgecolors="white", linewidths=1.5, zorder=3)
        tag = LANG_LABELS[lang] + (" (ring)" if LANG_RING[lang] else "")
        ax.annotate(tag, (iso, coh), xytext=(8, -2), textcoords="offset points",
                    fontsize=9, color=LANG_COLORS[lang], va="center")
    ax.axvline(0, color="0.7", lw=0.9, ls="--")
    ax.axhline(0.8, color="0.7", lw=0.9, ls=":")
    ax.set_xlabel("geometric path isometry (Pearson r)")
    ax.set_ylabel("steering coherence")
    ax.set_xlim(-0.3, 1.25)
    ax.set_ylim(0.70, 1.02)
    ax.set_title("Geometry vs steering quality", pad=6)
    save(fig, "fig7_isometry_coherence")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Cross-lingual steering coherence
# ══════════════════════════════════════════════════════════════════════════════
def fig8_steering_coherence():
    targets = ["zh", "ja", "fr"]
    cross_keys = {"zh": ("en", "zh"), "ja": ("zh", "ja"), "fr": ("en", "fr")}
    cross = {("en", "zh"): (0.9631, 0.0005),
             ("zh", "ja"): (0.7701, 0.0235),
             ("en", "fr"): (0.9802, 0.0004)}
    same = {"en": 0.7829, "zh": 0.9664, "ja": 0.9785, "fr": 0.7445}

    x = np.arange(len(targets)); w = 0.38
    base = [same[t] for t in targets]
    cmean = [cross[cross_keys[t]][0] for t in targets]
    cerr = [cross[cross_keys[t]][1] for t in targets]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - w/2, base, w*0.9, label="Same-language baseline",
           color=GRAY, edgecolor="white", lw=0.5)
    ax.bar(x + w/2, cmean, w*0.9, yerr=cerr, capsize=3, label="Cross-lingual steering",
           color=CORAL, edgecolor="white", lw=0.5,
           error_kw=dict(ecolor="#9c4a36", lw=1))
    for j, t in enumerate(targets):
        src = cross_keys[t][0]
        ax.text(j + w/2, cmean[j] + cerr[j] + 0.02,
                f"{src.upper()}→{t.upper()}\n{cmean[j]:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=CORAL)
    ax.set_xticks(x); ax.set_xticklabels([LANG_LABELS[t] for t in targets])
    ax.set_ylabel("geometric steering coherence")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Cross-lingual steering coherence", pad=6)
    ax.text(0.98, 0.04,
            "EN ring steers FR better than FR's own paths\n(only 2.3% subspace overlap)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555555", style="italic")
    save(fig, "fig8_steering_coherence")


if __name__ == "__main__":
    print(f"Saving figures to: {FIGURES_DIR}")
    fig1_language_manifolds()
    fig2_joint_embedding()
    fig3_subspace_alignment()
    fig4_overlap_heatmap()
    fig5_pullback_comparison()
    fig6_safety_projection()
    fig7_isometry_coherence()
    fig8_steering_coherence()
    print("All figures saved.")
