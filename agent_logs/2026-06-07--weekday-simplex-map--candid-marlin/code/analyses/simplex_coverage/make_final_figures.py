"""Final publication-grade figure set for the session conclusions (offline; local artifacts only).

  F1  steering head-to-head: E_BC by method x layer x concept, with Wilcoxon significance
  F2  two-regime geometry: chart residual + entropy vs distance-from-anchors (all concepts)
      + weekday L31 foliation with the core zoom (the 'valid blob hides a chart' figure)
  F3  ring skeletons: argmax-centroid cycles in their principal plane (weekdays/months/hues12)
  F4  steering to unprompted colours: achieved P vs natural ceiling per held colour (hybrid)
  F5  ablation robustness: leave-region-out across all concepts + titration + multi-region
  F6  where to steer: E_BC by layer (methods) + natural-state isometry by depth

Outputs -> result/figures/F{1..6}_*.png/pdf. Run from the session root.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUT = os.path.join(S, "result", "figures")
os.makedirs(OUT, exist_ok=True)
ART = lambda c: os.path.join(S, "artifacts", f"{c}_simplex", "llama31_8b", "simplex_coverage")

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

MCOLORS = {"linear_full": "#888888", "manifold": "#1f77b4", "subspace_inv_full": "#ff9d4d",
           "subspace_inv_resid": "#d62728", "linear_subspace": "#bbbbbb"}
MLABEL = {"linear_full": "linear (paper Eq.1)", "manifold": "spline manifold (paper Eq.2)",
          "subspace_inv_full": "subspace inverse (full)", "subspace_inv_resid": "subspace inverse (resid-preserving)",
          "linear_subspace": "linear in subspace"}


def load_steer(concept, root):
    layers = {}
    for fp in glob.glob(os.path.join(ART(root), f"steer_compare_{concept}*.json")):
        layers.update(json.load(open(fp))["layers"])
    return layers


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------- F1 head-to-head
def f1():
    concepts = [("weekdays", "weekday"), ("months", "months")]
    hues = load_steer("hues12", "hues12")
    if hues:
        concepts.append(("hues12", "hues12"))
    fig, axes = plt.subplots(1, len(concepts), figsize=(6.2 * len(concepts), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (concept, root) in zip(axes, concepts):
        layers = load_steer(concept, root)
        Ls = sorted(layers, key=int)
        xs = np.arange(len(Ls))
        w = 0.16
        for j, m in enumerate(MCOLORS):
            mv = [layers[L][m]["ebc_sum"]["mean"] for L in Ls]
            se = [layers[L][m]["ebc_sum"]["se"] for L in Ls]
            ax.bar(xs + (j - 2) * w, mv, w, yerr=se, capsize=2, color=MCOLORS[m],
                   label=MLABEL[m] if concept == "weekdays" else None)
        # significance stars: ours vs manifold per layer
        for i, L in enumerate(Ls):
            ours = np.array([r["ebc_sum"] for r in layers[L]["subspace_inv_resid"]["per_pair"]])
            base = np.array([r["ebc_sum"] for r in layers[L]["manifold"]["per_pair"]])
            try:
                _, p = wilcoxon(ours, base)
            except ValueError:
                continue
            top = max(layers[L][m]["ebc_sum"]["mean"] for m in MCOLORS)
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))
            better = ours.mean() < base.mean()
            ax.text(i, top * 1.04, star if better else f"({star})", ha="center", fontsize=9,
                    color="k" if better else "#1f77b4")
        ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in Ls])
        ax.set_title(f"{concept}  (E_BC to $M_y$, lower = more natural)")
        ax.set_ylabel("cumulative energy $E_{BC}$" if concept == "weekdays" else "")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Steering head-to-head on identical endpoints & carriers — "
                 "subspace steering beats linear AND spline-manifold at L23+ (stars: Wilcoxon ours vs manifold)",
                 fontsize=11, y=1.04)
    save(fig, "F1_steering_headtohead")


# ---------------------------------------------------------------- F2 two-regime geometry
def f2():
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.32, wspace=0.3)
    # top row: regime curves per concept (residual left axis; L31 entropy right axis)
    for ci, (concept, root) in enumerate([("weekdays", "weekday"), ("months", "months"),
                                          ("hues12", "hues12")]):
        ax = fig.add_subplot(gs[0, ci])
        ga = json.load(open(os.path.join(ART(root), "geometry_analysis.json")))
        ax2 = ax.twinx()
        handles = []
        for L, ls in (("23", "--"), ("31", "-")):
            reg = ga["layers"].get(L, {}).get("chart_breakdown_by_distance", [])
            if not reg:
                continue
            x = [min(np.mean([float(v) for v in r["nn_range"].split("-")]), 11) for r in reg]
            h, = ax.plot(x, [r["chart_resid_med"] for r in reg], "o" + ls, color="#d62728", ms=4,
                         label=f"chart residual L{L}")
            handles.append(h)
            if L == "31":
                h2, = ax2.plot(x, [r["entropy_med"] for r in reg], "s-", color="#1f77b4", ms=4,
                               alpha=0.75, label="achieved entropy L31")
                handles.append(h2)
        ax.axvspan(0, 2, color="green", alpha=0.08)
        ax.text(1.0, 0.02, "calibrated core", ha="center", fontsize=8, color="green",
                transform=ax.get_xaxis_transform())
        ax.set_xlabel("distance from anchors (median NN spacings)")
        ax.set_ylim(0, None); ax2.set_ylim(0, None)
        if ci == 0:
            ax.set_ylabel("affine-chart residual", color="#d62728")
            ax.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=False)
        if ci == 2:
            ax2.set_ylabel("entropy (nats)", color="#1f77b4")
        else:
            ax2.set_yticklabels([])
        ax.set_title(concept)
    # bottom row: weekday L31 foliation — full box + core zoom + entropy zoom
    z = np.load(os.path.join(ART("weekday"), "geometry_L31.npz"))
    zc = np.load(os.path.join(ART("weekday"), "geometry_core_L31.npz"))
    C = np.concatenate([z["samp_coords"], zc["samp_coords"]], 0).astype(float)
    D = np.concatenate([z["samp_dists"], zc["samp_dists"]], 0).astype(float)
    V = np.concatenate([z["valid"], zc["valid"]], 0).astype(bool)
    AC, AD = z["anchor_coords"].astype(float), z["anchor_dists"].astype(float)
    n = AD.shape[1] - 1
    Cv, Dv = C[V], D[V]
    amax_v = Dv[:, :n].argmax(1)
    amax_a = AD[:, :n].argmax(1)
    cents = np.stack([AC[amax_a == d].mean(0) for d in range(n)], 0)
    cmap = plt.get_cmap("tab10")
    titles = ["full box: argmax sectors ('all valid' view)",
              "core zoom: foliation + ring skeleton",
              "core zoom: achieved entropy"]
    lims = [None, 3.0, 3.0]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pw = np.sqrt(((AC[:, None, :] - AC[None, :, :]) ** 2).sum(-1))
    NN = float(np.median(np.sort(pw, 1)[:, 1]))
    for pi in range(3):
        ax = fig.add_subplot(gs[1, pi])
        if lims[pi] is None:
            sel = np.ones(len(Cv), bool)
        else:
            r0 = np.abs(Cv[:, :2] - AC[:, :2].mean(0)).max(1)
            sel = r0 < lims[pi] * NN
        if pi < 2:
            for d in range(n):
                s2 = sel & (amax_v == d)
                ax.scatter(Cv[s2, 0], Cv[s2, 1], s=2 if pi == 0 else 5, color=cmap(d), alpha=0.35)
        else:
            ent = -(Dv[:, :n] * np.log(np.clip(Dv[:, :n], 1e-12, None))).sum(1)
            sc = ax.scatter(Cv[sel, 0], Cv[sel, 1], s=5, c=ent[sel], cmap="viridis", alpha=0.6)
            plt.colorbar(sc, ax=ax, label="entropy (nats)", shrink=0.8)
        if pi > 0:
            ax.scatter(AC[:, 0], AC[:, 1], s=14, facecolors="none", edgecolors="k", lw=0.5)
            ring = np.concatenate([cents[:, :2], cents[:1, :2]], 0)
            ax.plot(ring[:, 0], ring[:, 1], "k-o", lw=1.4, ms=5)
            if pi == 1:
                for d in range(n):
                    ax.annotate(days[d], cents[d, :2], fontsize=8, fontweight="bold",
                                xytext=(3, 3), textcoords="offset points")
        else:
            bx = AC[:, :2].mean(0)
            rect = plt.Rectangle(bx - 3 * NN, 6 * NN, 6 * NN, fill=False, color="k", lw=1.4, ls="--")
            ax.add_patch(rect)
        ax.set_title(titles[pi], fontsize=9.5)
        ax.set_xlabel("subspace dim 1"); ax.set_ylabel("dim 2" if pi == 0 else "")
    fig.suptitle("Two-regime geometry: a calibrated affine core (the paper's ring is its skeleton) "
                 "inside a saturated 'valid' shell — weekdays L31", fontsize=12, y=0.97)
    save(fig, "F2_two_regime_geometry")


# ---------------------------------------------------------------- F3 ring skeletons
def f3():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (concept, root, L, names) in zip(axes, [
            ("weekdays Δ⁷", "weekday", 31, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            ("months Δ¹²", "months", 31, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
            ("hue wheel Δ¹²", "hues12", 31, ["Red", "Orange", "Yellow", "Lime", "Green", "Teal", "Cyan", "Azure", "Blue", "Violet", "Purple", "Pink"])]):
        z = np.load(os.path.join(ART(root), f"geometry_L{L}.npz"))
        AC, AD = z["anchor_coords"].astype(float), z["anchor_dists"].astype(float)
        n = AD.shape[1] - 1
        amax = AD[:, :n].argmax(1)
        ok = [d for d in range(n) if (amax == d).any()]
        cents = np.stack([AC[amax == d].mean(0) for d in ok], 0)
        cc = cents - cents.mean(0)
        _, _, Vt = np.linalg.svd(cc, full_matrices=False)
        c2 = cc @ Vt[:2].T
        a2 = (AC - cents.mean(0)) @ Vt[:2].T
        cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
        for i, d in enumerate(ok):
            ax.scatter(a2[amax == d, 0], a2[amax == d, 1], s=8, color=cmap(d % 20), alpha=0.22)
        order = np.argsort(np.arctan2(c2[:, 1], c2[:, 0]))
        ring = np.concatenate([c2[order], c2[order][:1]], 0)
        ax.plot(ring[:, 0], ring[:, 1], "k-", lw=1.3, alpha=0.7)
        ax.scatter(c2[:, 0], c2[:, 1], s=120, c=[cmap(d % 20) for d in ok], edgecolors="k",
                   linewidths=1.2, zorder=5)
        for i, d in enumerate(ok):
            ax.annotate(names[d], c2[i], fontsize=9.5, fontweight="bold", xytext=(5, 5),
                        textcoords="offset points")
        ga = json.load(open(os.path.join(ART(root), "geometry_analysis.json")))
        match = ga["layers"][str(L)]["ring"]["circular_order_match"]
        lim = np.abs(c2).max() * 1.7
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{concept} — cyclic order {match} (L{L})")
    fig.suptitle("The concept cycles in activation space: argmax-conditional anchor centroids in their principal plane "
                 "(line = angular order). Trained temporal cycles ring perfectly; the colour wheel only partially.",
                 fontsize=11, y=1.03)
    save(fig, "F3_ring_skeletons")


# ---------------------------------------------------------------- F4 unprompted steering
def f4():
    ceil = {c: v["ceiling_all_prompts"] for c, v in json.load(open(os.path.join(
        ART("colors"), "color_ceilings.json")))["ceilings"].items()}
    d = json.load(open(os.path.join(ART("colors"), "steer_unprompted_RGB_L27pca_hybrid.json")))
    per = d["per_held"]
    names = sorted(per, key=lambda h: -per[h]["P_gradient"])
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(names))
    ax.bar(x, [per[h]["P_gradient"] for h in names], 0.62, color="#d62728", label="steered P (hybrid, RGB-only chart)")
    ax.plot(x, [ceil.get(h, 0) for h in names], "k_", ms=13, mew=2, label="natural ceiling (max P over 255 prompts)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=8)
    for i, h in enumerate(names):
        if per[h]["P_gradient"] >= 0.5 and ceil.get(h, 0) < 0.5:
            ax.annotate("▲", (i, per[h]["P_gradient"] + 0.02), ha="center", fontsize=8, color="#d62728")
    ax.set_ylabel("P(held colour)")
    ax.set_title(f"Steering to colours NEVER in the anchor set — chart from {d['keep']} anchors only, L{d['layer']} "
                 f"(▲ = steered ABOVE the natural ceiling)")
    ax.legend(fontsize=9, frameon=False)
    # weekday + hue sanity panels
    for gi, (concept, root, tag) in enumerate([("weekdays", "weekday", "MonThu_L31pca_hybrid"),
                                               ("hues12", "hues12", "RGB_L31pca_hybrid")]):
        axd = fig.add_subplot(gs[1, gi])
        dd = json.load(open(os.path.join(ART(root), f"steer_unprompted_{tag}.json")))
        hn = list(dd["per_held"])
        axd.bar(np.arange(len(hn)), [dd["per_held"][h]["P_gradient"] for h in hn], 0.55, color="#d62728")
        axd.axhline(0.5, color="gray", ls=":", lw=1)
        axd.set_xticks(np.arange(len(hn))); axd.set_xticklabels(hn, rotation=45, fontsize=8)
        axd.set_ylim(0, 1.05)
        axd.set_title(f"{concept}: keep {dd['keep']} → steer the rest", fontsize=9.5)
        axd.set_ylabel("P(held)" if gi == 0 else "")
    save(fig, "F4_unprompted_steering")


# ---------------------------------------------------------------- F5 ablations
def f5():
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    # weekdays: 7 single + 2 multi + titration inset as extra bars
    ax = fig.add_subplot(gs[0, 0])
    items = []
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        for cand in (f"recovery_test_{day}.json", f"recovery_test_{day}_L31.json"):
            p = os.path.join(ART("weekday"), cand)
            if os.path.exists(p):
                items.append((day, json.load(open(p))["region_holdout"]["cell_recovery_rate"])); break
    for tag, lab in (("SatSun", "Sat+Sun"), ("MonTueWed", "Mon+Tue+Wed\n(55% anchors)")):
        p = os.path.join(ART("weekday"), f"recovery_test_{tag}.json")
        items.append((lab, json.load(open(p))["region_holdout"]["cell_recovery_rate"]))
    x = np.arange(len(items))
    ax.bar(x, [v for _, v in items], color=["#1f77b4"] * 7 + ["#ff9d4d"] * 2)
    ax.set_xticks(x); ax.set_xticklabels([k for k, _ in items], rotation=45, fontsize=8, ha="right")
    ax.set_ylim(0, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("region cell-recovery")
    ax.set_title("weekdays: 7/7 regions + multi-region", fontsize=10)
    # months 12/12
    ax = fig.add_subplot(gs[0, 1])
    mo = []
    for f in sorted(glob.glob(os.path.join(ART("months"), "recovery_test_*_k13.json"))):
        d = json.load(open(f)); mo.append((d["holdout"], d["region_holdout"]["cell_recovery_rate"]))
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mo.sort(key=lambda t: order.index(t[0]))
    ax.bar(np.arange(len(mo)), [v for _, v in mo], color="#1f77b4")
    ax.set_xticks(np.arange(len(mo))); ax.set_xticklabels([k for k, _ in mo], rotation=45, fontsize=8)
    ax.set_ylim(0, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_title("months: 12/12 regions (k=13)", fontsize=10)
    # colour families: per-deleted-colour max P
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for tag in ("BlueFam_k24", "RedFam_k24"):
        d = json.load(open(os.path.join(ART("colors"), f"recovery_test_{tag}.json")))
        for cname, v in d["region_holdout"]["max_P_per_deleted"].items():
            rows.append((cname, v, tag.startswith("Blue")))
    x = np.arange(len(rows))
    ax.bar(x, [v for _, v, _ in rows], color=["#1f77b4" if b else "#d62728" for _, _, b in rows])
    ax.set_xticks(x); ax.set_xticklabels([k for k, _, _ in rows], rotation=90, fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("colours: delete whole blue/red FAMILIES —\nhue giants recover, shades don't (ceilings)")
    ax.set_ylabel("max P(deleted colour) recovered")
    fig.suptitle("Leave-region-out robustness: the map extrapolates to deleted regions across all concepts",
                 fontsize=12, y=1.04)
    save(fig, "F5_ablation_robustness")


# ---------------------------------------------------------------- F6 where to steer
def f6():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for concept, root, marker in [("weekdays", "weekday", "o"), ("months", "months", "s")]:
        layers = load_steer(concept, root)
        Ls = sorted(layers, key=int)
        for m in ("linear_full", "manifold", "subspace_inv_resid"):
            axes[0].plot([int(L) for L in Ls], [layers[L][m]["ebc_sum"]["mean"] for L in Ls],
                         marker + "-", color=MCOLORS[m], alpha=1.0 if concept == "weekdays" else 0.45,
                         label=MLABEL[m] if concept == "weekdays" else None)
        ga = json.load(open(os.path.join(ART(root), "geometry_analysis.json")))
        Ls2 = sorted(ga["layers"], key=int)
        axes[1].plot([int(L) for L in Ls2], [ga["layers"][L]["isometry_r_anchors"] for L in Ls2],
                     marker + "-", color="#2ca02c", alpha=1.0 if concept == "weekdays" else 0.45,
                     label=f"{concept}")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("E_BC (lower = more natural)")
    axes[0].set_title("steering naturalness by depth — best window L23–L27\n(solid: weekdays, faint: months)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("isometry r (anchor pairs)")
    axes[1].set_title("natural-state isometry strengthens with depth")
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "F6_where_to_steer")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5(); f6()
    print("all final figures written to result/figures/")
