import json
import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import pearsonr

BASE = "artifacts/natural_domains_arithmetic/gemma3_27b"
# 6 prior (vivid-marlin) + 4 new (this session)
SRCS = ["vi", "sw", "ja", "fr", "zh", "ko", "es", "hi", "id", "tr"]
NEW = {"es", "hi", "id", "tr"}


def dirname(s):
    return "weekdays" if s == "en" else f"weekdays_{s}"


def load_rot(d):
    p = f"{BASE}/{d}/{d}/subspace/pca_k64/result/rotation.safetensors"
    with safe_open(p, framework="pt") as f:
        return f.get_tensor("rotation_matrix").float().numpy()  # (D, k)


Qen = load_rot("weekdays")


def overlap_to_en(Qs):
    # principal-angle cosines = singular values of Qen^T Qs (orthonormal PCA columns)
    s = np.linalg.svd(Qen.T @ Qs, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s ** 2))  # mean cos^2  (matches vivid-marlin overlap defn)


def rj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# sanity: overlap(en,en) == 1.0, overlap(en,random) ~ k/D
print(f"[sanity] overlap(en,en)={overlap_to_en(Qen):.4f}  k/D={Qen.shape[1]/Qen.shape[0]:.4f}")

rows = []
for s in SRCS:
    d = dirname(s)
    md = rj(f"{BASE}/weekdays/weekdays/cross_lingual_steering/"
            f"natural_domains_arithmetic_weekdays_{s}/pca_k64/L54_last_token/spline_s0.0/metadata.json")
    transfer = md["coherence"]["mean"] if md else float("nan")
    ps = f"{BASE}/{d}/{d}/path_steering/pca_k64/L54_last_token/spline_s0.0/result/criteria"
    cj = rj(f"{ps}/coherence/geometric/metrics.json")
    own = cj["mean"] if cj else float("nan")
    ij = rj(f"{ps}/isometry/geometric/metrics.json")
    iso = ij["pearson_r"] if ij else float("nan")
    ov = overlap_to_en(load_rot(d))
    rows.append((s, transfer, ov, own, iso))

print(f"\n{'src':>4} {'new':>4} {'transfer':>9} {'overlap':>8} {'own_coh':>8} {'isometry':>9}")
for s, t, ov, own, iso in rows:
    iso_s = "nan" if (iso != iso) else f"{iso:.4f}"
    print(f"{s:>4} {('*' if s in NEW else ''):>4} {t:9.4f} {ov:8.4f} {own:8.4f} {iso_s:>9}")

T = np.array([r[1] for r in rows])
preds = {"overlap": np.array([r[2] for r in rows]),
         "own_coh": np.array([r[3] for r in rows]),
         "isometry": np.array([r[4] for r in rows])}

print("\n=== Pearson correlation of X->EN transfer vs predictor (n=10) ===")
for name, arr in preds.items():
    m = np.isfinite(T) & np.isfinite(arr)
    r, p = pearsonr(T[m], arr[m])
    print(f"  transfer ~ {name:8s}: r={r:+.3f}  p={p:.3f}  n={int(m.sum())}")

# also report prior-n=6-only for direct comparison with vivid-marlin
print("\n=== same, prior 6 only (vi,sw,ja,fr,zh,ko) ===")
idx6 = [i for i, r in enumerate(rows) if r[0] not in NEW]
T6 = T[idx6]
for name, arr in preds.items():
    a6 = arr[idx6]
    m = np.isfinite(T6) & np.isfinite(a6)
    r, p = pearsonr(T6[m], a6[m])
    print(f"  transfer ~ {name:8s}: r={r:+.3f}  p={p:.3f}  n={int(m.sum())}")
