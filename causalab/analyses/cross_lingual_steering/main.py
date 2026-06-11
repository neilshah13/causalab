"""Cross-lingual steering analysis.

Tests whether a source language's ring manifold (PCA + spline featurizer) can steer
a target language model's outputs. The featurizer is loaded from SOURCE task artifacts;
the model is loaded for the TARGET task. Outputs are measured by day-index shift.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    _task_config_for_metadata,
    resolve_task,
    generate_datasets,
    build_targets_for_grid,
)
from causalab.io.pipelines import load_pipeline
from causalab.tasks.loader import load_task_counterfactuals

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "cross_lingual_steering"


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run cross-lingual steering: source manifold steers target language prompts."""
    analysis = cfg[ANALYSIS_NAME]

    source_task = analysis.source_task
    source_experiment_root = analysis.source_experiment_root
    subspace_sub = analysis.subspace_sub
    layer = int(analysis.layer)
    token_position = analysis.token_position
    manifold_sub = analysis.manifold_sub
    n_eval_samples = int(analysis.n_eval_samples)
    n_prompts = int(analysis.n_prompts)
    num_steps = int(analysis.num_steps_along_path)
    batch_size = int(analysis.batch_size)
    max_pairs = analysis.get("max_pairs")

    root = cfg.experiment_root

    # ------------------------------------------------------------------
    # 1. Load TARGET task + model pipeline
    # ------------------------------------------------------------------
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )

    if task.intervention_variable is None:
        logger.warning("No intervention_variable on target task; aborting.")
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}

    # ------------------------------------------------------------------
    # 2. Build interchange target for the target model
    # ------------------------------------------------------------------
    targets, _tp_list = build_targets_for_grid(
        pipeline,
        task,
        [layer],
        [token_position],
    )
    token_pos = _tp_list[0]
    interchange_target = next(iter(targets.values()))

    # ------------------------------------------------------------------
    # 3. Load SOURCE featurizer into the target model's interchange hook
    # ------------------------------------------------------------------
    from causalab.analyses.activation_manifold.loading import load_featurizer

    manifold_out = os.path.join(
        source_experiment_root,
        "activation_manifold",
        subspace_sub,
        manifold_sub,
    )
    featurizer = load_featurizer(
        manifold_out, interchange_target, layer, token_pos.id
    )

    # ------------------------------------------------------------------
    # 4. Load TARGET raw features and project into SOURCE PCA space
    # ------------------------------------------------------------------
    from safetensors.torch import load_file as _load_file

    tv = cfg.task.get("target_variable")
    target_subspace_dir = os.path.join(root, "subspace", subspace_sub)
    if tv:
        target_subspace_dir = os.path.join(target_subspace_dir, tv)

    # Features are stored per cell under layer_x_pos/ (grid mode) or flat (single mode)
    from causalab.io.pipelines import load_subspace_metadata

    ss_meta = load_subspace_metadata(root, subspace_sub, target_variable=tv)
    _ss_mode = ss_meta.get("mode", "single")
    _ss_method = ss_meta.get("method", "pca")
    if _ss_mode == "grid" and _ss_method == "pca":
        _feat_dir = os.path.join(
            target_subspace_dir,
            "layer_x_pos",
            f"L{layer}_{token_position}",
            "features",
        )
    else:
        _feat_dir = os.path.join(target_subspace_dir, "features")

    raw_path = os.path.join(_feat_dir, "raw_features.safetensors")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Target raw_features.safetensors not found at {raw_path}. "
            "Re-run the subspace analysis for the target task to generate it."
        )
    raw_features = _load_file(raw_path)["features"]  # (N, D)

    # Apply SOURCE PCA transform to project target raw features into source PCA space.
    # The source featurizer stages: [0]=subspace (PCA), [1]=standardize, [2]=spline.
    from causalab.neural.pipeline import resolve_device

    device = torch.device(resolve_device(cfg.model.device))

    src_subspace_featurizer = featurizer.stages[0].featurizer
    with torch.no_grad():
        raw_on_device = raw_features.to(device)
        # Apply source PCA (subspace featurizer) — handles weight matrix projection
        if hasattr(src_subspace_featurizer, "transform"):
            pca_features = src_subspace_featurizer.transform(raw_on_device)
        elif hasattr(src_subspace_featurizer, "encode"):
            pca_features, _ = src_subspace_featurizer.encode(raw_on_device)
        else:
            # Fallback: direct weight matrix multiply (typical for linear PCA stage)
            W = src_subspace_featurizer.weight  # (k, D)
            pca_features = (raw_on_device - src_subspace_featurizer.mean) @ W.T

    # ------------------------------------------------------------------
    # 5. Compute TARGET task centroids in SOURCE manifold's coordinate system
    # ------------------------------------------------------------------
    values = task.intervention_values

    # Load training dataset aligned with features (use subspace saved dataset)
    saved_ds_path = os.path.join(target_subspace_dir, "train_dataset.json")
    if os.path.exists(saved_ds_path):
        from causalab.io.counterfactuals import load_counterfactual_examples

        train_ds = load_counterfactual_examples(saved_ds_path, task.causal_model)
    else:
        train_ds, _ = generate_datasets(
            task,
            n_train=ss_meta.get("n_train", cfg.task.get("n_train", 1000)),
            n_test=0,
            seed=ss_meta.get("seed", cfg.seed),
            enumerate_all=cfg.task.enumerate_all,
            resample_variable=cfg.task.get("resample_variable", "all"),
        )

    n_values = len(values)
    pca_centroids = torch.zeros(n_values, pca_features.shape[1], device=device)
    centroid_mask = torch.zeros(n_values, dtype=torch.bool)
    counts = torch.zeros(n_values)

    for i, ex in enumerate(train_ds[: pca_features.shape[0]]):
        ci = task.intervention_value_index(ex)
        pca_centroids[ci] += pca_features[i].to(device)
        counts[ci] += 1
    for ci in range(n_values):
        if counts[ci] > 0:
            pca_centroids[ci] /= counts[ci]
            centroid_mask[ci] = True

    # Encode PCA centroids through source standardize + spline to get intrinsic coords
    has_manifold = len(featurizer.stages) >= 2 and hasattr(
        featurizer.stages[-1].featurizer, "manifold"
    )
    manifold_obj = None
    if has_manifold:
        manifold_obj = featurizer.stages[-1].featurizer.manifold.to(device)
        std_stage = featurizer.stages[-2].featurizer
        feat_mean = std_stage._mean.to(device)
        feat_std = std_stage._std.to(device)
        standardized = (pca_centroids - feat_mean) / (feat_std + 1e-6)
        with torch.no_grad():
            spline_centroids, _ = manifold_obj.encode(standardized)
    else:
        logger.warning(
            "Source featurizer has no manifold stage; falling back to PCA centroids."
        )
        spline_centroids = pca_centroids

    # ------------------------------------------------------------------
    # 6. Generate TARGET evaluation samples
    # ------------------------------------------------------------------
    cf_mod = load_task_counterfactuals(task.name)
    filtered_samples = cf_mod.generate_dataset(
        task.causal_model,
        n_eval_samples,
        cfg.seed + 100,
    )
    eval_samples = filtered_samples[:n_prompts]

    # ------------------------------------------------------------------
    # 7. Build pairs and collect output distributions on target prompts
    # ------------------------------------------------------------------
    from causalab.methods.metric import tokenize_variable_values

    var_indices = tokenize_variable_values(
        pipeline.tokenizer,
        values,
        task.result_token_pattern,
    )

    all_pairs = list(itertools.combinations(range(n_values), 2))
    if max_pairs is not None and len(all_pairs) > int(max_pairs):
        import random as _random

        rng = _random.Random(cfg.seed)
        pairs = sorted(rng.sample(all_pairs, int(max_pairs)))
    else:
        pairs = sorted(all_pairs)

    logger.info(
        "Cross-lingual steering: %d pair(s) selected (%d total possible)",
        len(pairs),
        len(all_pairs),
    )

    from causalab.analyses.path_steering.path_mode import _build_geodesic_path

    pair_dists_list = []
    computed_pairs = []

    for si, ei in pairs:
        if not centroid_mask[si] or not centroid_mask[ei]:
            logger.warning("Skipping pair (%d, %d): missing centroid", si, ei)
            continue

        # Build geodesic path in SOURCE intrinsic space
        if manifold_obj is not None:
            grid_points = _build_geodesic_path(
                spline_centroids[si],
                spline_centroids[ei],
                num_steps,
                manifold_obj,
            )
        else:
            # Linear path in PCA space (fallback when no manifold)
            alphas = torch.linspace(0, 1, num_steps, device=device)
            grid_points = torch.stack(
                [
                    (1 - a) * spline_centroids[si] + a * spline_centroids[ei]
                    for a in alphas
                ]
            )

        from causalab.methods.steer.collect import collect_grid_distributions

        probs = collect_grid_distributions(
            pipeline=pipeline,
            grid_points=grid_points,
            interchange_target=interchange_target,
            filtered_samples=eval_samples,
            var_indices=var_indices,
            batch_size=batch_size,
            n_base_samples=n_prompts,
            average=False,
            full_vocab_softmax=True,
        )
        pair_dists_list.append(probs)
        computed_pairs.append((si, ei))

    # ------------------------------------------------------------------
    # 8. Compute coherence score
    # ------------------------------------------------------------------
    from causalab.methods.scores.coherence import compute_score as _coherence_score

    coherence_result: dict[str, float] = {"mean": float("nan"), "se": float("nan")}
    pair_distributions = None

    if pair_dists_list:
        pair_distributions = torch.stack(pair_dists_list)  # (n_pairs, num_steps, n_prompts, W)
        logger.info(
            "Collected cross-lingual pair distributions: %s",
            pair_distributions.shape,
        )
        coherence_result = _coherence_score(
            pair_distributions=pair_distributions,
            output_dir=None,
            path_mode_label="geometric",
        )
    else:
        logger.warning("No pair distributions collected; coherence is NaN.")

    # ------------------------------------------------------------------
    # 9. Save metadata.json
    # ------------------------------------------------------------------
    out_dir = os.path.join(
        root,
        "cross_lingual_steering",
        source_task,
        subspace_sub,
        f"L{layer}_{token_position}",
        manifold_sub,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save pair distributions alongside metadata
    if pair_distributions is not None:
        from safetensors.torch import save_file as _save_file

        _save_file(
            {"pair_distributions": pair_distributions.float()},
            os.path.join(out_dir, "pair_distributions.safetensors"),
        )
        with open(os.path.join(out_dir, "pairs.json"), "w") as f:
            json.dump(
                {
                    "pairs": computed_pairs,
                    "values": [str(v) for v in values],
                    "num_steps": num_steps,
                },
                f,
                indent=2,
            )

    metadata: dict[str, Any] = {
        "analysis": ANALYSIS_NAME,
        "source_task": source_task,
        "source_experiment_root": source_experiment_root,
        "target_task": cfg.task.name,
        "target_task_config": _task_config_for_metadata(
            OmegaConf.to_container(cfg.task, resolve=True)
        ),
        "model": cfg.model.name,
        "subspace_sub": subspace_sub,
        "layer": layer,
        "token_position": token_position,
        "manifold_sub": manifold_sub,
        "n_eval_samples": n_eval_samples,
        "n_prompts": n_prompts,
        "num_steps_along_path": num_steps,
        "batch_size": batch_size,
        "max_pairs": max_pairs,
        "n_pairs_computed": len(computed_pairs),
        "coherence": coherence_result,
        "seed": cfg.seed,
    }

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Cross-lingual steering complete. coherence=%.4f. Artifacts: %s",
        coherence_result.get("mean", float("nan")),
        out_dir,
    )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata
