"""Pure functions for cross-lingual manifold alignment."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def compute_raw_centroids(
    raw_features: Tensor,
    train_dataset: list,
    task,
) -> Tensor:
    """Average raw activations per intervention value class.

    Args:
        raw_features: (N, D) tensor of raw activations.
        train_dataset: list of counterfactual examples, row-aligned with raw_features.
        task: causalab task with .intervention_values and .intervention_value_index().

    Returns:
        (n_values, D) tensor of per-class mean activations.
    """
    n_values = len(task.intervention_values)
    D = raw_features.shape[1]
    centroids = torch.zeros(n_values, D, device=raw_features.device)
    counts = torch.zeros(n_values, device=raw_features.device)
    for i, ex in enumerate(train_dataset[: raw_features.shape[0]]):
        ci = task.intervention_value_index(ex)
        centroids[ci] += raw_features[i]
        counts[ci] += 1
    for ci in range(n_values):
        if counts[ci] > 0:
            centroids[ci] /= counts[ci]
    return centroids


def _fit_2d_basis(centroids: Tensor) -> np.ndarray:
    """Return a (2, D) orthonormal basis spanning the best 2D fit of the centroids."""
    c = centroids.numpy().astype(np.float64)
    c = c - c.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return Vt[:2]  # (2, D)


def compute_subspace_alignment(
    centroids_a: Tensor,
    centroids_b: Tensor,
) -> dict:
    """Compute principal angles between the 2D ring subspaces of two centroid sets.

    Both centroid sets should be in the same ambient space (same D).

    Returns dict with:
        principal_angle_1_deg, principal_angle_2_deg: smaller and larger angle (degrees)
        subspace_overlap_score: mean squared cosine of principal angles in [0, 1].
            1.0 = identical subspaces, 0.0 = orthogonal.
        cos_theta_1, cos_theta_2: cosines of principal angles.
    """
    basis_a = _fit_2d_basis(centroids_a)  # (2, D)
    basis_b = _fit_2d_basis(centroids_b)  # (2, D)
    M = basis_a @ basis_b.T               # (2, 2) cross-Gram matrix
    _, sigma, _ = np.linalg.svd(M)
    sigma = np.clip(sigma, 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(sigma))
    return {
        "principal_angle_1_deg": float(angles_deg[0]),
        "principal_angle_2_deg": float(angles_deg[1]),
        "subspace_overlap_score": float((sigma ** 2).mean()),
        "cos_theta_1": float(sigma[0]),
        "cos_theta_2": float(sigma[1]),
    }


def fit_circle_2d(
    points: Tensor,
) -> tuple[np.ndarray, float, float]:
    """Fit a circle to 2D points using algebraic least squares (Coope 1993).

    Returns:
        center: (2,) array [cx, cy]
        radius: float
        residual_mse: mean squared distance from each point to the fitted circle
    """
    pts = points.numpy().astype(np.float64)
    n = pts.shape[0]
    B = np.column_stack([2 * pts, np.ones(n)])
    d = (pts ** 2).sum(axis=1)
    result, _, _, _ = np.linalg.lstsq(B, d, rcond=None)
    cx, cy = result[0], result[1]
    center = np.array([cx, cy])
    r_sq = result[2] + cx ** 2 + cy ** 2
    radius = float(np.sqrt(max(r_sq, 0.0)))
    dists = np.sqrt(((pts - center) ** 2).sum(axis=1))
    residuals = dists - radius
    mse = float((residuals ** 2).mean())
    return center, radius, mse


def compute_joint_mse_ratio(
    centroids_a: Tensor,
    centroids_b: Tensor,
) -> dict:
    """Measure whether two centroid sets lie on the same 2D ring.

    Projects both centroid sets into a shared 2D subspace (PCA on the
    joint set), fits circles to each independently, and computes the
    ratio of the larger fitted radius to the smaller one.

    When both sets lie on the same ring, their fitted radii in the
    shared projection are nearly identical (ratio ~1).  When they lie
    on orthogonal rings, the joint PCA plane causes one set to project
    to a drastically larger or smaller apparent radius (ratio >> 1).

    Returns dict with:
        joint_mse: MSE of a single circle fit to all points jointly.
        mse_a, mse_b: per-set circle-fit MSE.
        mse_ratio: max(r_a, r_b) / min(r_a, r_b) — the key
            discriminator; ~1 = same ring, >> 1 = different rings.
        radius_a, radius_b, radius_joint: fitted radii (sanity check).
    """
    from sklearn.decomposition import PCA

    joint = torch.cat([centroids_a, centroids_b], dim=0).numpy().astype(np.float64)
    mean = joint.mean(axis=0)
    pca = PCA(n_components=2)
    pca.fit(joint - mean)

    proj_a = torch.from_numpy(pca.transform(centroids_a.numpy() - mean)).float()
    proj_b = torch.from_numpy(pca.transform(centroids_b.numpy() - mean)).float()
    proj_joint = torch.cat([proj_a, proj_b], dim=0).float()

    _, r_a, mse_a = fit_circle_2d(proj_a)
    _, r_b, mse_b = fit_circle_2d(proj_b)
    _, r_joint, mse_joint = fit_circle_2d(proj_joint)

    # Radius ratio: robust to near-zero individual MSE, separates same vs
    # different rings clearly (orthogonal rings project to wildly different radii).
    r_max = max(r_a, r_b)
    r_min = min(r_a, r_b)
    ratio = r_max / (r_min + 1e-10)

    return {
        "joint_mse": float(mse_joint),
        "mse_a": float(mse_a),
        "mse_b": float(mse_b),
        "mse_ratio": float(ratio),
        "radius_a": float(r_a),
        "radius_b": float(r_b),
        "radius_joint": float(r_joint),
    }
