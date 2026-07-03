"""Assets for the final slide deck (weekdays only). Outputs -> result/slides/.

Styled per PLOT_STYLE.md: semantic palette (CORAL = ours/highlight, CHAR = the paper's method,
GRAY = weak baseline, TEAL = good/faithful, GOLD = spare 4th accent), twilight for per-day
colours, white-edged circle markers, short bold titles, png + pdf.

  S1a  the manifold is built from 7 centroid points only (what's drawn vs what's computed)
  S1b  ring schematic: manifold vs linear vs centroid-to-centroid vectors + evidence bars
  S2   method flow diagram (boxes + arrows)
  S3   naturalness bars, layers 23+27 (L31 omitted: E_BC is blind to teleporting there —
       linear vs manifold n.s. p~0.18; see REPORT Step 9)
  S4   recovery graph, anchors = Mon+Thu only, all five held days recovered
  S5   steered-text examples card

Run from session root: python3 code/analyses/simplex_coverage/make_slide_assets.py
"""

from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
})

TEAL = "#2A9D8F"
CORAL = "#E76F51"
GRAY = "#B0BEC5"
CHAR = "#264653"
GOLD = "#E9C46A"
# twilight sampled at half-steps so day 0 isn't the colormap's near-white endpoint
DC = [plt.cm.twilight((i + 0.5) / 7) for i in range(7)]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
W = os.path.join(S, "artifacts", "weekday_simplex", "llama31_8b", "simplex_coverage")
OLD = os.path.join(S, "..", "2026-06-03--manifold-vs-vector--lucid-geode",
                   "artifacts", "natural_domains_arithmetic_weekdays", "llama31_8b", "weekdays")
OUT = os.path.join(S, "result", "slides")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), facecolor="white")
    plt.close(fig)
    print("saved", name)


def day_dot(ax, xy, d, s=200):
    ax.scatter(*xy, marker="o", s=s, color=DC[d], edgecolors="white", linewidths=1.5, zorder=5)


def centroid_ring():
    z = np.load(os.path.join(W, "geometry_L31.npz"))
    AC, AD = z["anchor_coords"].astype(float), z["anchor_dists"].astype(float)
    amax = AD[:, :7].argmax(1)
    cents = np.stack([AC[amax == d].mean(0) for d in range(7)], 0)
    cc = cents - cents.mean(0)
    _, _, Vt = np.linalg.svd(cc, full_matrices=False)
    return cc @ Vt[:2].T


def smooth_closed(P, n=400):
    from scipy.interpolate import CubicSpline
    x = np.arange(len(P) + 1)
    cs = CubicSpline(x, np.vstack([P, P[:1]]), axis=0, bc_type="periodic")
    return cs(np.linspace(0, len(P), n))


# ---------------------------------------------------------------- S1a
def s1a():
    C = centroid_ring()
    order = np.argsort(np.arctan2(C[:, 1], C[:, 0]))
    ring = smooth_closed(C[order])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    ax = axes[0]
    ax.plot(ring[:, 0], ring[:, 1], "-", color=CHAR, lw=5, alpha=0.9, zorder=2)
    ax.plot(ring[:, 0], ring[:, 1], "-", color=CHAR, lw=14, alpha=0.12, zorder=1)
    for d in range(7):
        day_dot(ax, C[d], d, s=190)
        ax.annotate(DAYS[d][:3], C[d], xytext=(9, 7), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=CHAR,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_title("What the paper draws", pad=6)
    ax = axes[1]
    for d in range(7):
        day_dot(ax, C[d], d, s=190)
        ax.annotate(DAYS[d][:3], C[d], xytext=(9, 7), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=CHAR,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.plot(ring[:, 0], ring[:, 1], "--", color=GRAY, lw=1.6, zorder=1)
    ro = C[order]
    for i in range(7):
        mid = (ro[i] + ro[(i + 1) % 7]) / 2 * 1.18
        ax.text(*mid, "?", fontsize=17, color=CORAL, ha="center", va="center", fontweight="bold")
    ax.set_title("What is actually computed", pad=6)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        m = np.abs(C).max() * 1.45
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    axes[0].text(0.5, -0.04, "a smooth curve over the whole region",
                 transform=axes[0].transAxes, ha="center", fontsize=9.5, color="#555555")
    axes[1].text(0.5, -0.04, "7 averaged points + a fitting step — the space between is assumed",
                 transform=axes[1].transAxes, ha="center", fontsize=9.5, color="#555555")
    save(fig, "S1a_manifold_is_centroids")


# ---------------------------------------------------------------- S1b
def s1b():
    th = np.pi / 2 - np.arange(7) * 2 * np.pi / 7
    ro = np.stack([np.cos(th), np.sin(th)], 1)
    circ = np.stack([np.cos(np.linspace(0, 2 * np.pi, 400)),
                     np.sin(np.linspace(0, 2 * np.pi, 400))], 1)
    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(False)
    ax.plot(circ[:, 0], circ[:, 1], "-", color="0.85", lw=1.4, zorder=1)
    a, b = ro[0], ro[6]
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=GRAY, lw=6, zorder=2)
    ax.annotate("Linear: straight\nto the endpoint", np.array([-1.92, 1.30]),
                fontsize=11, color=GRAY, fontweight="bold", ha="left")
    arc_t = np.linspace(th[0], th[0] - 6 * 2 * np.pi / 7, 220)
    ax.plot(np.cos(arc_t), np.sin(arc_t), "-", color=CHAR, lw=6, zorder=3, solid_capstyle="round")
    mid34 = (th[3] + th[4]) / 2
    ax.annotate("Manifold:\nthrough the week", 1.52 * np.array([np.cos(mid34), np.sin(mid34)]),
                fontsize=11, color=CHAR, fontweight="bold", ha="center")
    for i in range(6):
        ax.add_patch(FancyArrowPatch(ro[i] * 0.84, ro[i + 1] * 0.84, arrowstyle="-|>",
                                     mutation_scale=22, color=CORAL, lw=3.0, zorder=4,
                                     shrinkA=7, shrinkB=7))
    ax.annotate("vectors,\ncentroid → centroid", (0, -0.04), fontsize=11,
                color=CORAL, fontweight="bold", ha="center")
    for d in range(7):
        day_dot(ax, ro[d], d, s=210)
        ax.annotate(DAYS[d][:3], ro[d] * 1.21, fontsize=10, ha="center", fontweight="bold",
                    color=CHAR, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.8); ax.set_ylim(-1.75, 1.62)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Steering Monday → Sunday", pad=6)
    fid = json.load(open(os.path.join(OLD, "steering_compare", "default", "metrics",
                                      "fidelity_at_intermediate.json")))
    days = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(days))
    w = 0.27
    series = (("chord", "Linear (one straight vector)", GRAY),
              ("polyline", "vectors centroid → centroid", CORAL),
              ("spline", "fitted manifold", CHAR))
    for j, (m_, lab, col) in enumerate(series):
        vals = [fid[m_][d]["p_correct"] for d in days]
        ax2.bar(x + (j - 1) * w, vals, w * 0.9, color=col, label=lab,
                edgecolor="white", linewidth=0.5)
        if col == CORAL:
            for xi, v in zip(x, vals):
                ax2.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color=CORAL)
    ax2.set_xticks(x); ax2.set_xticklabels([d[:3] for d in days])
    ax2.set_ylabel("P(intended day) at each waypoint")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right")
    ax2.set_title("Vectors match the manifold", pad=6)
    ax2.text(0.5, -0.17,
             "at the endpoints (Mon, Sun) all three methods inject the same centroid state and succeed equally —\n"
             "the difference is only in what happens between the points",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="#555555")
    save(fig, "S1b_vectors_match_manifold")


# ---------------------------------------------------------------- S2 flow
def s2():
    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    ax.axis("off")
    boxes = [
        (0.02, '194 anchor prompts\n\n“A day that is not Monday:”\n“The day after Friday:”\n“A weekend day:” …',
         "#FDF2E3"),
        (0.265, "model activations\n\none 4096-number state\nper prompt (layer 23–31)", "#EAF2FB"),
        (0.51, "keep the ~6 directions\nthat matter for days\n\n(PCA / correlation\nwith behaviour)", "#EAF2FB"),
        (0.755, "walk around the anchors\n\nforce nearby states,\nread the answer\nthat comes out", "#FBEAEA"),
    ]
    for x0, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x0, 0.18), 0.20, 0.62,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    facecolor=col, edgecolor=CHAR, lw=1.6))
        ax.text(x0 + 0.10, 0.49, txt, ha="center", va="center", fontsize=10.5, color=CHAR)
    for x0 in (0.222, 0.467, 0.712):
        ax.add_patch(FancyArrowPatch((x0, 0.49), (x0 + 0.04, 0.49), arrowstyle="-|>",
                                     mutation_scale=22, color="#555555", lw=2.2))
    ax.set_xlim(-0.005, 0.975); ax.set_ylim(0.05, 0.95)
    save(fig, "S2_method_flow")


# ---------------------------------------------------------------- S3 bars (L31 omitted)
def s3():
    d = json.load(open(os.path.join(W, "steer_compare_weekdays.json")))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sel = [("linear_full", "goodfire: straight vector", GRAY),
           ("manifold", "goodfire: fitted manifold", CHAR),
           ("subspace_inv_resid", "ours: discovered map", CORAL)]
    Ls = ["23", "27"]                       # L31 omitted: E_BC is blind to teleporting there
    x = np.arange(len(Ls))
    w = 0.26
    for j, (m, lab, col) in enumerate(sel):
        vals = [d["layers"][L][m]["ebc_sum"]["mean"] for L in Ls]
        ax.bar(x + (j - 1) * w, vals, w * 0.9,
               yerr=[d["layers"][L][m]["ebc_sum"]["se"] for L in Ls], capsize=3,
               color=col, label=lab, edgecolor="white", linewidth=0.5)
        if col == CORAL:
            for xi, v in zip(x, vals):
                ax.text(xi + w, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=CORAL)
    ax.set_xticks(x); ax.set_xticklabels([f"layer {L}" for L in Ls], fontweight="bold")
    ax.set_ylabel("distance from natural behaviour\n(the paper's score; lower = better)")
    ax.legend(loc="upper left")
    ax.set_title("Steering naturalness (lower = better)", pad=6)
    ax.text(0.5, -0.16,
            "identical start/end points and prompts; ours vs either baseline p < 10⁻¹². The last layer is omitted:\n"
            "there the score stops separating linear from manifold (n.s., p≈0.18) because it measures distance\n"
            "to natural behaviour, not smoothness — teleporting linear paths sit near their natural endpoints.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    save(fig, "S3_naturalness_bars")


# ---------------------------------------------------------------- S4 recovery
def s4():
    z = np.load(os.path.join(W, "geometry_keepMonThu_L31.npz"))
    C, D, V = z["samp_coords"].astype(float), z["samp_dists"].astype(float), z["valid"].astype(bool)
    AC, AD, kept = (z["anchor_coords"].astype(float), z["anchor_dists"].astype(float),
                    z["kept_mask"].astype(bool))
    n = 7
    Dv = D[V]
    amax_s = Dv[:, :n].argmax(1)
    sqa = np.sqrt(np.clip(AD, 0, None)); mA = sqa.mean(0)
    _, _, Vt = np.linalg.svd(sqa - mA, full_matrices=False); comps = Vt[:2]
    BA = (sqa - mA) @ comps.T
    BS = (np.sqrt(np.clip(Dv, 0, None)) - mA) @ comps.T
    held_days = [d for d in range(n) if d not in (0, 3)]
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    base = np.isin(amax_s, [0, 3])
    ax.scatter(BS[base, 0], BS[base, 1], s=4, color=TEAL, alpha=0.3,
               label="walked samples landing on the prompted days (Mon/Thu)")
    rec = ~base
    ax.scatter(BS[rec, 0], BS[rec, 1], s=6, color=CORAL, alpha=0.4,
               label="recovered samples — the five NEVER-PROMPTED days")
    ax.scatter(BA[kept, 0], BA[kept, 1], s=70, color=CHAR, marker="o",
               edgecolors="white", linewidths=1.2,
               label="training anchors (Monday + Thursday — the ONLY prompts used)")
    ax.scatter(BA[~kept, 0], BA[~kept, 1], s=64, facecolors="none", edgecolors=GOLD,
               linewidths=2.0, label="true positions of the deleted days' anchors")
    for d in held_days:
        sel = AD[:, :n].argmax(1) == d
        cent = BA[sel].mean(0)
        ax.annotate(DAYS[d][:3], cent, fontsize=13, fontweight="bold", color=CHAR,
                    xytext=(6, 6), textcoords="offset points",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    ax.set_xlabel("Hellinger-PC1"); ax.set_ylabel("Hellinger-PC2")
    ax.set_title("Two anchors recover the week", pad=6)
    ax.text(0.5, 1.012, "anchors: Monday + Thursday only — every deleted day's region is reached, P(day) up to 1.0",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
    save(fig, "S4_recover_all_days")


# ---------------------------------------------------------------- S5 steered-text card
def s5():
    d = json.load(open(os.path.join(W, "steer_gen_demo_L23.json")))["results"]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axis("off")
    y = 0.97
    ax.text(0.0, y, "Steering our map, then letting the model keep writing",
            fontsize=13.5, fontweight="bold", color=CHAR, va="top")
    y -= 0.07
    ax.text(0.0, y, "Same prompt, only the internal state is changed — the model lands on the "
                    "steered day at P ≈ 0.99 and the story follows it. (layer 23, greedy decoding)",
            fontsize=10, color="#555555", va="top")
    y -= 0.09
    entries = [
        ("UNSTEERED (baseline)", d[0]["carrier"], d[0]["unsteered"]["text"], None,
         f"day-word mass leader: {d[0]['unsteered']['top_day']} at only P = {d[0]['unsteered']['P']}"),
        ("steered → Wednesday", d[0]["carrier"], d[0]["Wed"]["text"], "Wednesday",
         f"P(Wednesday) = {d[0]['Wed']['P_target']}"),
        ("steered → Saturday", d[1]["carrier"], d[1]["Sat"]["text"], "Saturday",
         f"P(Saturday) = {d[1]['Sat']['P_target']}"),
        ("steered → Sunday", d[2]["carrier"], d[2]["Sun"]["text"], "Sunday",
         f"P(Sunday) = {d[2]['Sun']['P_target']}"),
    ]
    for label, carrier, text, day, stat in entries:
        color = GRAY if day is None else CORAL
        ax.text(0.0, y, label, fontsize=10.5, fontweight="bold", color=color, va="top")
        ax.text(0.62, y, stat, fontsize=9.5, color="#555555", va="top")
        y -= 0.045
        full = (carrier + text).replace("<|end_of_text|>", "").strip()
        wrapped = textwrap.fill("“" + full + "…”", 124)
        ax.text(0.015, y, wrapped, fontsize=10, style="italic", color="#333333", va="top",
                linespacing=1.25)
        y -= 0.052 * (wrapped.count("\n") + 1) + 0.052
    save(fig, "S5_steered_text_examples")


if __name__ == "__main__":
    s1a(); s1b(); s2(); s3(); s4(); s5()
    print("slide assets ->", OUT)
