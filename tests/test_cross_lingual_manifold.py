"""Tests for cross_lingual_manifold core functions."""
import math
import pytest
import torch
import numpy as np
from causalab.analyses.cross_lingual_manifold.core import (
    compute_subspace_alignment,
    fit_circle_2d,
    compute_joint_mse_ratio,
    compute_raw_centroids,
)


def _ring_centroids(n: int, dim: int, plane: tuple[int, int] = (0, 1)) -> torch.Tensor:
    """Generate n centroids on a unit circle in the given two dimensions of a dim-D space."""
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:n]
    c = torch.zeros(n, dim)
    c[:, plane[0]] = torch.cos(angles)
    c[:, plane[1]] = torch.sin(angles)
    return c


def test_subspace_alignment_identical():
    """Identical centroid sets → principal angle near 0, overlap ≈ 1."""
    c = _ring_centroids(7, 64)
    m = compute_subspace_alignment(c, c)
    assert m["principal_angle_1_deg"] < 2.0
    assert m["subspace_overlap_score"] > 0.98


def test_subspace_alignment_orthogonal():
    """Rings in orthogonal planes → largest principal angle near 90."""
    c_a = _ring_centroids(7, 64, plane=(0, 1))
    c_b = _ring_centroids(7, 64, plane=(2, 3))
    m = compute_subspace_alignment(c_a, c_b)
    assert m["principal_angle_2_deg"] > 80.0
    assert m["subspace_overlap_score"] < 0.1


def test_fit_circle_2d_perfect_ring():
    """Points on a unit circle → residual MSE near 0."""
    angles = torch.linspace(0, 2 * math.pi, 8)[:7]
    pts = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    center, radius, mse = fit_circle_2d(pts)
    assert abs(radius - 1.0) < 0.01
    assert mse < 1e-6


def test_joint_mse_ratio_same_ring():
    """Same ring (plus tiny noise) → ratio close to 1."""
    c = _ring_centroids(7, 64)
    noise = 0.01 * torch.randn_like(c)
    ratio_info = compute_joint_mse_ratio(c, c + noise)
    assert ratio_info["mse_ratio"] < 1.5


def test_joint_mse_ratio_different_rings():
    """Two rings in orthogonal planes → ratio >> 1."""
    c_a = _ring_centroids(7, 64, plane=(0, 1))
    c_b = _ring_centroids(7, 64, plane=(2, 3))
    ratio_info = compute_joint_mse_ratio(c_a, c_b)
    assert ratio_info["mse_ratio"] > 3.0


def test_compute_raw_centroids_shape():
    """compute_raw_centroids returns (n_values, D) tensor."""
    raw_features = torch.randn(70, 32)

    class FakeTask:
        intervention_values = list(range(7))
        def intervention_value_index(self, ex):
            return ex["label"]

    class FakeEx:
        def __init__(self, lbl):
            self.data = {"label": lbl}
        def __getitem__(self, k):
            return self.data[k]

    dataset = [FakeEx(i % 7) for i in range(70)]
    task = FakeTask()
    centroids = compute_raw_centroids(raw_features, dataset, task)
    assert centroids.shape == (7, 32)
