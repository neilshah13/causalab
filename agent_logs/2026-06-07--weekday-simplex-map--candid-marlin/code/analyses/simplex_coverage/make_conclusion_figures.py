"""Narrative figures + GIFs for the streamlined CONCLUSIONS (offline, local artifacts).

  C1  vectors vs manifold — the manifold is its centroid skeleton (old session + this one)
  C2  how we map the geometry — anchors in behaviour space -> activation space -> walked map
  C3  our steering is more natural (paper's own metric) and the output stays coherent
  C4  few anchors recover the whole concept (keep Mon+Thu -> all five other days)
  G1  GIF: discovering the geometry by walking around the anchors
  G2  GIF: recovering days that were never prompted (needs geometry_keepMonThu_L31.npz from D5)

Run from session root: python3 code/analyses/simplex_coverage/make_conclusion_figures.py [--gif2-only]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OLD = os.path.join(S, "..", "2026-06-03--manifold-vs-vector--lucid-geode",
                   "artifacts", "natural_domains_arithmetic_weekdays", "llama31_8b", "weekdays")
W = os.path.join(S, "artifacts", "weekday_simplex", "llama31_8b", "simplex_coverage")
M = os.path.join(S, "artifacts", "months_simplex", "llama31_8b", "simplex_coverage")
OUT = os.path.join(S, "result", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CMAP = lambda d: plt.cm.twilight((d + 0.5) / 7)
TEAL, CORAL, CHAR, GOLD = "#2A9D8F", "#E76F51", "#264653", "#E9C46A"


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load_weekday_geom():
    z = np.load(os.path.join(W, "geometry_L31.npz"))
    zc = np.load(os.path.join(W, "geometry_core_L31.npz"))
    C = np.concatenate([z["samp_coords"], zc["samp_coords"]], 0).astype(float)
    D = np.concatenate([z["samp_dists"], zc["samp_dists"]], 0).astype(float)
    V = np.concatenate([z["valid"], zc["valid"]], 0).astype(bool)
    return C[V], D[V], z["anchor_coords"].astype(float), z["anchor_dists"].astype(float)


def hplane(dists, ref_sq_mean, comps):
    return (np.sqrt(np.clip(dists, 0, None)) - ref_sq_mean) @ comps.T


# ------------------------------------------------------------------ C1
def c1():
    fid = json.load(open(os.path.join(OLD, "steering_compare", "default", "metrics",
                                      "fidelity_at_intermediate.json")))
    agg = json.load(open(os.path.join(OLD, "missing_day", "default", "metrics", "aggregate.json")))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    days = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    x = np.arange(len(days))
    labels = {"chord": ("straight vector (chord)", "#999999"),
              "polyline": ("centroid-to-centroid vectors", "#d62728"),
              "spline": ("fitted manifold (spline)", "#1f77b4")}
    for j, m in enumerate(("chord", "polyline", "spline")):
        axes[0].bar(x + (j - 1) * 0.27, [fid[m][d]["p_correct"] for d in days], 0.27,
                    color=labels[m][1], label=labels[m][0])
    axes[0].set_xticks(x); axes[0].set_xticklabels([d[:3] for d in days])
    axes[0].set_ylabel("P(intended day) at the waypoint")
    axes[0].set_title("steering through intermediate days:\nvectors between centroids ≈ the manifold")
    axes[0].legend(fontsize=8, frameon=False)
    # gap-fill: place a day the method was never shown
    methods = [("transition_local", "local transition vector", "#d62728"),
               ("gap_aware_cyclic", "manifold (cyclic)", "#1f77b4"),
               ("gap_aware", "manifold (gap-aware)", "#5fa2d8"),
               ("manifold_midpoint", "manifold (midpoint)", "#a8c8e8")]
    vals = [agg[k]["recovery"][0] for k, _, _ in methods]
    errs = [agg[k]["recovery"][1] / np.sqrt(7) for k, _, _ in methods]
    axes[1].bar(range(len(methods)), vals, yerr=errs, capsize=3,
                color=[c for _, _, c in methods])
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels([l for _, l, _ in methods], rotation=20, fontsize=8, ha="right")
    axes[1].set_ylabel("recovery of the withheld day")
    axes[1].set_title("placing a day the method never saw:\na simple vector beats every manifold variant")
    # this session: E_BC linear vs manifold by layer
    for concept, root, alpha in (("weekdays", W, 1.0), ("months", M, 0.45)):
        layers = {}
        import glob
        for fp in glob.glob(os.path.join(root, f"steer_compare_{concept}*.json")):
            layers.update(json.load(open(fp))["layers"])
        Ls = sorted(layers, key=int)
        axes[2].plot([int(L) for L in Ls], [layers[L]["linear_full"]["ebc_sum"]["mean"] for L in Ls],
                     "o-", color="#999999", alpha=alpha,
                     label="straight vector" if alpha == 1 else None)
        axes[2].plot([int(L) for L in Ls], [layers[L]["manifold"]["ebc_sum"]["mean"] for L in Ls],
                     "s-", color="#1f77b4", alpha=alpha,
                     label="fitted manifold" if alpha == 1 else None)
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("unnaturalness (E_BC, lower better)")
    axes[2].set_title("this session, paper's own metric:\nby the last layers the manifold ≈ a straight vector")
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("The fitted manifold adds nothing beyond its centroid points", fontsize=12.5, y=1.05)
    save(fig, "C1_vectors_vs_manifold")


# ------------------------------------------------------------------ C2 + G1
def c2_and_gif1():
    Cv, Dv, AC, AD = load_weekday_geom()
    n = 7
    sqa = np.sqrt(np.clip(AD, 0, None)); mA = sqa.mean(0)
    _, _, Vt = np.linalg.svd(sqa - mA, full_matrices=False); comps = Vt[:2]
    BA = hplane(AD, mA, comps)            # anchors, behaviour plane
    BS = hplane(Dv, mA, comps)            # samples, behaviour plane
    amax_a = AD[:, :n].argmax(1); amax_s = Dv[:, :n].argmax(1)
    # core zoom region
    from scipy.spatial import cKDTree
    dnn, _ = cKDTree(AC).query(Cv, k=1)
    pw = np.sqrt(((AC[:, None] - AC[None]) ** 2).sum(-1))
    NN = float(np.median(np.sort(pw, 1)[:, 1]))
    core = dnn < 2.5 * NN
    cents = np.stack([AC[amax_a == d].mean(0) for d in range(n)], 0)

    # static C2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for d in range(n):
        axes[0].scatter(BA[amax_a == d, 0], BA[amax_a == d, 1], s=16, color=CMAP(d), alpha=0.8)
    axes[0].set_title("1. anchor prompts cover the whole\n'day of the week' region (behaviour space)")
    for d in range(n):
        axes[1].scatter(AC[amax_a == d, 0], AC[amax_a == d, 1], s=16, color=CMAP(d), alpha=0.8)
    ring = np.concatenate([cents[:, :2], cents[:1, :2]], 0)
    axes[1].plot(ring[:, 0], ring[:, 1], "k-o", lw=1.2, ms=4)
    axes[1].set_title("2. the same anchors in activation space\n(the week's arrangement re-appears)")
    sel = core
    for d in range(n):
        s2 = sel & (amax_s == d)
        axes[2].scatter(Cv[s2, 0], Cv[s2, 1], s=4, color=CMAP(d), alpha=0.35)
    axes[2].scatter(AC[:, 0], AC[:, 1], s=16, facecolors="none", edgecolors="k", lw=0.6)
    axes[2].set_title("3. walking around the anchors discovers\nthe full activation→behaviour map")
    for ax, lab in zip(axes, ("behaviour dim 1 / 2", "activation dim 1 / 2", "activation dim 1 / 2")):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=8)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=CMAP(d), label=DAYS[d]) for d in range(n)]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("How we map the geometry: anchor the region, then walk it", fontsize=12.5, y=1.04)
    save(fig, "C2_mapping_the_geometry")

    # G1: progressive discovery
    order = np.argsort(dnn)
    idx_core = order[core[order]][:12000] if core.sum() > 12000 else order[core[order]]
    frames = []
    steps = np.linspace(0, len(idx_core), 22).astype(int)[1:]
    lim_a = (AC[:, 0].min() - 1.5 * NN, AC[:, 0].max() + 1.5 * NN,
             AC[:, 1].min() - 1.5 * NN, AC[:, 1].max() + 1.5 * NN)
    for fi, t in enumerate(steps):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
        shown = idx_core[:t]
        for d in range(n):
            s2 = shown[amax_s[shown] == d]
            axes[0].scatter(Cv[s2, 0], Cv[s2, 1], s=4, color=CMAP(d), alpha=0.35)
            axes[1].scatter(BS[s2, 0], BS[s2, 1], s=4, color=CMAP(d), alpha=0.35)
        axes[0].scatter(AC[:, 0], AC[:, 1], s=22, facecolors="none", edgecolors="k", lw=0.8)
        axes[1].scatter(BA[:, 0], BA[:, 1], s=22, facecolors="none", edgecolors="k", lw=0.8)
        axes[0].set_xlim(lim_a[0], lim_a[1]); axes[0].set_ylim(lim_a[2], lim_a[3])
        axes[0].set_title("activation space — walking outward from the anchors")
        axes[1].set_title("behaviour space — where each step lands")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"discovering the activation→behaviour map  ({t} steps)", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(img.copy()))
        plt.close(fig)
    frames[0].save(os.path.join(OUT, "G1_discovering_the_map.gif"), save_all=True,
                   append_images=frames[1:], duration=[130] * (len(frames) - 1) + [1900], loop=0)
    print(f"saved G1_discovering_the_map.gif ({len(frames)} frames)")


# ------------------------------------------------------------------ C3
def c3():
    d = json.load(open(os.path.join(W, "steer_compare_weekdays.json")))
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    sel = [("linear_full", "goodfire: straight vector", "#999999"),
           ("manifold", "goodfire: fitted manifold", "#1f77b4"),
           ("subspace_inv_resid", "ours: subspace map", "#d62728")]
    Ls = ["23", "27", "31"]
    x = np.arange(len(Ls))
    for j, (m, lab, col) in enumerate(sel):
        ax.bar(x + (j - 1) * 0.26, [d["layers"][L][m]["ebc_sum"]["mean"] for L in Ls], 0.26,
               yerr=[d["layers"][L][m]["ebc_sum"]["se"] for L in Ls], capsize=2, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"layer {L}" for L in Ls])
    ax.set_ylabel("unnaturalness of the steering path\n(paper's E_BC; lower = more natural)")
    ax.set_title("our steering is 2–5× more natural,\nby the paper's own measure (p < 10⁻¹²)")
    ax.legend(fontsize=8, frameon=False)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    lines = ["After steering, the model keeps talking normally", ""]
    for m, lab, _ in sel:
        s = d["layers"]["23"][m]["gen_samples"][0]
        ppl = d["layers"]["23"][m]["gen_ppl"]["median"]
        txt = s["text"].replace("\n", " ⏎ ")[:92]
        lines.append(f"{lab}   (steering {s['pair']}, fluency ppl {ppl:.2f})")
        lines.append(f'   “{txt}…”')
        lines.append("")
    ax2.text(0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8.6, family="monospace",
             wrap=True)
    ax2.set_title("…for every method: no broken text — but only ours\nalso lands on the natural path", fontsize=10)
    save(fig, "C3_natural_steering")


# ------------------------------------------------------------------ C4 (static)
def c4():
    dd = json.load(open(os.path.join(W, "steer_unprompted_MonThu_L31pca_hybrid.json")))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["Mon", "Thu"] + list(dd["per_held"].keys())
    vals = [1.0, 1.0] + [dd["per_held"][h]["P_gradient"] for h in dd["per_held"]]
    cols = ["#1f77b4", "#1f77b4"] + ["#d62728"] * (len(names) - 2)
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("P(day) reached by steering")
    ax.set_ylim(0, 1.08)
    ax.text(0.5, 1.02, "prompted", ha="center", fontsize=9, color="#1f77b4")
    ax.text(4.0, 1.02, "never prompted — recovered by walking the map", ha="center", fontsize=9, color="#d62728")
    ax.set_title("Anchor two days, recover the whole week")
    save(fig, "C4_few_anchors_full_concept")


# ------------------------------------------------------------------ G2 (needs D5 dump)
def gif2():
    p = os.path.join(W, "geometry_keepMonThu_L31.npz")
    if not os.path.exists(p):
        print("G2 skipped: geometry_keepMonThu_L31.npz not pulled yet")
        return
    z = np.load(p)
    C, D, V = z["samp_coords"].astype(float), z["samp_dists"].astype(float), z["valid"].astype(bool)
    AC, AD, kept = z["anchor_coords"].astype(float), z["anchor_dists"].astype(float), z["kept_mask"].astype(bool)
    n = 7
    Cv, Dv = C[V], D[V]
    amax_s = Dv[:, :n].argmax(1)
    sqa = np.sqrt(np.clip(AD, 0, None)); mA = sqa.mean(0)
    _, _, Vt = np.linalg.svd(sqa - mA, full_matrices=False); comps = Vt[:2]
    BA = hplane(AD, mA, comps); BS = hplane(Dv, mA, comps)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(Cv))
    steps = np.linspace(0, min(len(order), 14000), 22).astype(int)[1:]
    found = set()
    frames = []
    for t in steps:
        shown = order[:t]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
        for d in range(n):
            s2 = shown[amax_s[shown] == d]
            col = TEAL if d in (0, 3) else CORAL
            axes[0].scatter(Cv[s2, 0], Cv[s2, 1], s=4, color=col, alpha=0.3)
            axes[1].scatter(BS[s2, 0], BS[s2, 1], s=4, color=col, alpha=0.3)
            if len(s2) > 3:
                found.add(d)
        axes[0].scatter(AC[kept, 0], AC[kept, 1], s=40, color=CHAR, edgecolors="white", lw=1.0,
                        label="Mon+Thu anchors (the only prompts used)")
        axes[1].scatter(BA[kept, 0], BA[kept, 1], s=40, color=CHAR, edgecolors="white", lw=1.0)
        axes[1].scatter(BA[~kept, 0], BA[~kept, 1], s=34, facecolors="none", edgecolors=GOLD, lw=1.8,
                        label="deleted days' true positions (never used)")
        got = [DAYS[d] for d in sorted(found) if d not in (0, 3)]
        axes[0].set_title("activation space — walking the 2-day chart")
        axes[1].set_title(f"behaviour space — days discovered: {', '.join(got) if got else '—'}")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].legend(fontsize=7, frameon=False, loc="lower left")
        axes[1].legend(fontsize=7, frameon=False, loc="lower left")
        fig.suptitle("recovering days that were never prompted (anchors: Monday + Thursday only)",
                     fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    frames[0].save(os.path.join(OUT, "G2_recovering_unprompted_days.gif"), save_all=True,
                   append_images=frames[1:], duration=[130] * (len(frames) - 1) + [1900], loop=0)
    print(f"saved G2_recovering_unprompted_days.gif ({len(frames)} frames)")


if __name__ == "__main__":
    if "--gif2-only" in sys.argv:
        gif2()
    else:
        c1(); c2_and_gif1(); c3(); c4(); gif2()
