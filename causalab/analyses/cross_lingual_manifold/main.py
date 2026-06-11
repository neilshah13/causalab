"""Cross-lingual manifold alignment analysis.

Reads saved raw_features.safetensors from two experiment roots (target and
source) and computes principal angles between their 2D ring subspaces plus a
joint circle-fit MSE ratio.  No model weights are loaded.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.analyses.cross_lingual_manifold.core import (
    compute_joint_mse_ratio,
    compute_raw_centroids,
    compute_subspace_alignment,
)
from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.tasks.loader import load_task

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "cross_lingual_manifold"


def _load_centroids(
    experiment_root: str,
    task_name: str,
    task_config: dict,
    target_variable: str,
    subspace_sub: str,
    layer: int,
    token_position: str,
) -> torch.Tensor:
    """Load raw features and compute per-class centroids for one experiment root.

    Args:
        experiment_root: Root directory of the experiment.
        task_name: Name of the task (e.g. "natural_domains_arithmetic").
        task_config: Dict of task-specific config fields.
        target_variable: The intervention variable to use.
        subspace_sub: Subspace sub-directory name (e.g. "pca_k2").
        layer: Layer index.
        token_position: Token position name (e.g. "last_token").

    Returns:
        (n_values, D) tensor of per-class mean activations.

    Raises:
        FileNotFoundError: If raw_features.safetensors is not found.
    """
    from safetensors.torch import load_file

    # Build cell directory path
    cell_dir = os.path.join(
        experiment_root,
        "subspace",
        subspace_sub,
        "layer_x_pos",
        f"L{layer}_{token_position}",
    )

    features_path = os.path.join(cell_dir, "features", "raw_features.safetensors")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"raw_features.safetensors not found at {features_path}. "
            "Run the subspace analysis first to populate the features cache."
        )

    # Load raw features
    raw_features = load_file(features_path)["raw_features"]
    logger.info("Loaded raw features %s from %s", tuple(raw_features.shape), features_path)

    # Resolve dataset path: cell dir first, fall back to subspace root
    ds_path = os.path.join(cell_dir, "train_dataset.json")
    if not os.path.exists(ds_path):
        ds_path = os.path.join(
            experiment_root, "subspace", subspace_sub, "train_dataset.json"
        )
    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            f"train_dataset.json not found under {cell_dir} or its parent subspace dir."
        )

    # Load task (no model weights needed)
    task = load_task(task_name, task_cfg=task_config if task_config else None)
    task.intervention_variable = target_variable

    # Load dataset aligned with features
    train_dataset = load_counterfactual_examples(ds_path, task.causal_model)
    logger.info("Loaded dataset: %d examples", len(train_dataset))

    # Compute per-class centroids
    centroids = compute_raw_centroids(raw_features, train_dataset, task)
    logger.info("Computed centroids: %s", tuple(centroids.shape))
    return centroids


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the cross-lingual manifold alignment analysis.

    Reads pre-computed raw features from two experiment roots and returns
    alignment metrics without loading model weights.
    """
    analysis = cfg[ANALYSIS_NAME]

    source_task: str = analysis.source_task
    source_root: str = analysis.source_experiment_root
    subspace_sub: str = analysis.subspace_sub
    layer: int = int(analysis.layer)
    token_position: str = str(analysis.token_position)

    target_root: str = cfg.experiment_root

    # Resolve task config for the target task
    task_config = OmegaConf.to_container(cfg.task, resolve=True)
    task_name: str = cfg.task.name
    target_variable: str = cfg.task.get("target_variable")

    logger.info("=== Cross-lingual manifold alignment ===")
    logger.info("Target task: %s  root: %s", task_name, target_root)
    logger.info("Source task: %s  root: %s", source_task, source_root)
    logger.info(
        "Subspace: %s  layer: %d  position: %s", subspace_sub, layer, token_position
    )

    # Load centroids for target task
    target_centroids = _load_centroids(
        experiment_root=target_root,
        task_name=task_name,
        task_config=task_config,
        target_variable=target_variable,
        subspace_sub=subspace_sub,
        layer=layer,
        token_position=token_position,
    )

    # Build source task config: start from target config and override the name.
    # This ensures task-level fields like target_variable and domain settings
    # are inherited, so the source task is resolved with the same structure.
    source_task_config = dict(task_config)
    source_task_config["name"] = source_task

    source_centroids = _load_centroids(
        experiment_root=source_root,
        task_name=source_task,
        task_config=source_task_config,
        target_variable=target_variable,
        subspace_sub=subspace_sub,
        layer=layer,
        token_position=token_position,
    )

    # Compute alignment metrics
    alignment = compute_subspace_alignment(target_centroids, source_centroids)
    joint = compute_joint_mse_ratio(target_centroids, source_centroids)

    metrics = {
        "analysis": ANALYSIS_NAME,
        "target_task": task_name,
        "source_task": source_task,
        "subspace_sub": subspace_sub,
        "layer": layer,
        "token_position": token_position,
        "subspace_alignment": alignment,
        "joint_centroid_fit": joint,
    }

    # Write results
    out_dir = os.path.join(target_root, ANALYSIS_NAME, source_task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", out_path)

    return metrics
