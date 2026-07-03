"""Colour natural max-P ceilings + ceiling-normalized sparse-recovery rescoring (offline).

The colour sparse-anchor recovery is flat at ~10-19% across k=16-40 with threshold
P(colour) >= 0.5 — but that threshold silently assumes every colour CAN reach 0.5.
Many gamut colours (Beige, Tan, Silver, ...) may never get half the mass in ANY natural
context, in which case "not recovered" is a property of the model's output prior, not of
the map. This script measures each colour's natural ceiling = max P(colour) over all
captured colour prompts (255, incl. non-retained), then rescores the sparse runs with a
ceiling-normalized criterion: recovered* = max_P_steered >= 0.5 * ceiling.

Pure numpy + local artifacts; no GPU. Run from the session root:
  python3 code/analyses/simplex_coverage/color_ceilings.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
CROOT = os.path.join(SESSION, "artifacts", "colors_simplex", "llama31_8b", "simplex_coverage")

CANONICAL_COLORS = [
    "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Black", "White",
    "Gray", "Grey", "Cyan", "Magenta", "Violet", "Indigo", "Turquoise", "Teal", "Maroon", "Navy",
    "Olive", "Lime", "Aqua", "Gold", "Silver", "Beige", "Tan", "Peach", "Coral", "Salmon",
    "Crimson", "Scarlet", "Lavender", "Plum", "Mint", "Ivory", "Charcoal", "Amber", "Azure",
    "Rose", "Ruby", "Emerald", "Sapphire", "Jade", "Mustard", "Burgundy", "Mauve", "Khaki",
    "Cream", "Bronze", "Copper",
]


def main():
    from safetensors import safe_open
    cj = json.load(open(os.path.join(SESSION, "code", "methods", "color_prompts", "colors.json")))
    single = set(cj["single_token_colors"])
    tokens = [c for c in CANONICAL_COLORS if c in single]
    n = len(tokens)
    with safe_open(os.path.join(CROOT, "activations.safetensors"), framework="np") as f:
        dists = f.get_tensor("dists")
        retained = f.get_tensor("retained").astype(bool)
    assert dists.shape[1] == n + 1, (dists.shape, n)

    ceil_all = dists[:, :n].max(0)                       # over ALL 255 prompts
    ceil_ret = dists[retained][:, :n].max(0)             # over the 127 anchors
    argmax_counts = np.bincount(dists[retained][:, :n].argmax(1), minlength=n)

    ceilings = {tokens[i]: {
        "ceiling_all_prompts": round(float(ceil_all[i]), 3),
        "ceiling_anchors": round(float(ceil_ret[i]), 3),
        "n_anchors_argmax": int(argmax_counts[i]),
    } for i in range(n)}

    # rescore every sparse_recovery json with the ceiling-normalized criterion
    rescored = {}
    for path in sorted(glob.glob(os.path.join(CROOT, "sparse_recovery_*.json"))):
        d = json.load(open(path))
        name = os.path.basename(path)
        entry = {"keep": d["keep"], "results": []}
        for res in d["results"]:
            per = res.get("per_day", {})
            held = [h for h in d["held"] if h in per]
            raw_rec, norm_rec, unreachable = 0, 0, 0
            detail = {}
            for h in held:
                mp = per[h]["max_P"]
                ceil = ceil_all[tokens.index(h)] if h in tokens else 0.0
                raw = mp >= d.get("recover_thresh", 0.5)
                norm = (ceil > 0.05) and (mp >= 0.5 * ceil)
                low_ceil = ceil < 0.5
                raw_rec += raw; norm_rec += norm; unreachable += low_ceil
                detail[h] = {"max_P": mp, "ceiling": round(float(ceil), 3),
                             "ratio": round(float(mp / ceil), 3) if ceil > 0 else None,
                             "recovered_raw": bool(raw), "recovered_ceilnorm": bool(norm)}
            entry["results"].append({
                "method": res["method"], "k": res["k"], "n_held": len(held),
                "recovered_raw_frac": round(raw_rec / max(len(held), 1), 3),
                "recovered_ceilnorm_frac": round(norm_rec / max(len(held), 1), 3),
                "n_held_with_ceiling_below_0.5": int(unreachable),
                "per_day": detail,
            })
        rescored[name] = entry

    out = {"n_colors": n, "tokens": tokens,
           "n_colors_ceiling_below_0.5": int((ceil_all < 0.5).sum()),
           "n_colors_ceiling_below_0.25": int((ceil_all < 0.25).sum()),
           "ceilings": ceilings, "sparse_rescored": rescored}
    with open(os.path.join(CROOT, "color_ceilings.json"), "w") as f:
        json.dump(out, f, indent=2)

    # summary print
    order = np.argsort(ceil_all)
    print(f"colours with natural ceiling < 0.5: {int((ceil_all < 0.5).sum())}/{n}")
    print("lowest 15 ceilings:", {tokens[i]: round(float(ceil_all[i]), 2) for i in order[:15]})
    for name, entry in rescored.items():
        print(f"\n{name} (keep {entry['keep']}):")
        for r in entry["results"]:
            print(f"  {r['method']:>4} k={r['k']:<3} raw {r['recovered_raw_frac']:.0%} "
                  f"-> ceiling-normalized {r['recovered_ceilnorm_frac']:.0%} "
                  f"({r['n_held_with_ceiling_below_0.5']} held colours have ceiling<0.5)")

    # figure: ceilings bar + raw vs normalized recovery
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(n), ceil_all[order], color="tab:blue")
    axes[0].axhline(0.5, color="red", ls="--", lw=1, label="sparse 'recovered' threshold")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels([tokens[i] for i in order], rotation=90, fontsize=6)
    axes[0].set_ylabel("natural ceiling  max P(colour) over 255 prompts")
    axes[0].set_title("Many colours can never pass the 0.5 threshold naturally")
    axes[0].legend(fontsize=8)
    labels, raw_v, norm_v = [], [], []
    for name, entry in rescored.items():
        nm = "keep3" if "RedGreenBlue" in name and "Orange" not in name else "keep8"
        for r in entry["results"]:
            if r["method"] == "pca":
                labels.append(f"{nm}\nk={r['k']}")
                raw_v.append(r["recovered_raw_frac"])
                norm_v.append(r["recovered_ceilnorm_frac"])
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, raw_v, 0.4, label="raw (P>=0.5)")
    axes[1].bar(x + 0.2, norm_v, 0.4, label="ceiling-normalized")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("frac held-out colours recovered")
    axes[1].set_title("Sparse recovery rescored against natural ceilings (pca)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_dir = os.path.join(CROOT, "figures"); os.makedirs(fig_dir, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"color_ceilings.{e}"))
    print("\nsaved -> color_ceilings.json + figures/color_ceilings.png")


if __name__ == "__main__":
    main()
