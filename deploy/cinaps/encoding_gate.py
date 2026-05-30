#!/usr/bin/env python3
"""Self-contained Stage-0 encoding gate for cyclic-arithmetic tasks.

The shipped ``baseline`` analysis scores with ``generated == raw_output``
exact match at ``max_new_tokens: 1`` — which makes any *multi-token* answer
unscorable (e.g. French ``mercredi`` -> ``[' mer','cre','di']``). For the
multilingual cycles that means the gate measures tokenization, not knowledge
(FR weekdays: 0/7 answers are single-token).

This script fixes that the simple way: generate a few tokens, then do a
**lenient first-word match**. It reuses causalab's task / dataset / pipeline
primitives and writes ``{experiment_root}/encoding_gate/accuracy.json``.

Usage::

    uv run python deploy/cinaps/encoding_gate.py \
        --task natural_domains_arithmetic_weekdays_fr --model llama31_8b
    uv run python deploy/cinaps/encoding_gate.py --task ... --dry-run   # no model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata


def words(s: str) -> list[str]:
    """Lowercased runs of unicode letters, in order."""
    return [w.lower() for w in re.findall(r"[^\W\d_]+", s, flags=re.UNICODE)]


def first_word(s: str) -> str:
    ws = words(s)
    return ws[0] if ws else s.strip().lower()


def deaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def predict(generated: str, vocab: set[str], vocab_da: dict[str, str]) -> str:
    """First valid *cycle word* in the generation — article-proof.

    Models answer with a leading determiner ("Le vendredi", "El sábado") or
    restate ("El día es sábado"); a plain first-word match grabs the article
    and undercounts. Scanning for the first word that is an actual answer
    candidate fixes that while still failing genuine non-answers ("Quel jour
    est trois…"). Falls back to the raw first word so misses stay misses."""
    for w in words(generated):
        if w in vocab:
            return w
        da = deaccent(w)
        if da in vocab_da:
            return vocab_da[da]
    return first_word(generated)


def build_cfg(task: str, model: str, max_new_tokens: int, target_variable: str, experiment_root: str | None):
    import causalab
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from causalab.io.configs import apply_experiment_root_variant

    shipped = os.path.join(os.path.dirname(causalab.__file__), "configs")
    overrides = [
        f"task={task}",
        f"model={model}",
        f"task.max_new_tokens={max_new_tokens}",
        f"+task.target_variable={target_variable}",  # '+' : key not in task schema (runners add it)
    ]
    if experiment_root:
        overrides.append(f"experiment_root={experiment_root}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=shipped, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    apply_experiment_root_variant(cfg)  # appends task.variant to experiment_root
    return cfg


def load_dataset(cfg):
    from omegaconf import OmegaConf

    from causalab.runner.helpers import generate_datasets, resolve_task

    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    train_dataset, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )
    return task, train_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="task config name, e.g. natural_domains_arithmetic_weekdays_fr")
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--target-variable", default="result", help="variable to score (calendar cycles use 'result')")
    ap.add_argument("--experiment-root", default=None, help="base root; task variant is appended")
    ap.add_argument("--gate", type=float, default=0.60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="build cfg + dataset and preview prompts; no model load")
    args = ap.parse_args()

    cfg = build_cfg(args.task, args.model, args.max_new_tokens, args.target_variable, args.experiment_root)
    task, dataset = load_dataset(cfg)
    out_dir = os.path.join(cfg.experiment_root, "encoding_gate")

    if args.dry_run:
        print(f"task={args.task} model={args.model} n_examples={len(dataset)}")
        print(f"experiment_root={cfg.experiment_root}")
        print(f"out_dir={out_dir}")
        print("--- sample (prompt -> expected) ---")
        for ex in dataset[:8]:
            raw_in = ex["input"].get("raw_input")
            raw_out = ex["input"]["raw_output"]
            print(f"  {raw_in!r}  ->  {raw_out!r}  (first_word={first_word(raw_out if isinstance(raw_out, str) else raw_out[0])})")
        return

    os.makedirs(out_dir, exist_ok=True)
    from causalab.io.pipelines import load_pipeline

    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=args.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )

    # Candidate cycle words (the 7/12 valid answers) — used to skip articles.
    vocab: set[str] = set()
    for ex in dataset:
        raw = ex["input"]["raw_output"]
        for a in raw if isinstance(raw, list) else [raw]:
            vocab.add(first_word(a))
    vocab_da = {deaccent(w): w for w in vocab}

    correct = correct_da = total = 0
    per_class: dict[str, dict[str, int]] = {}
    examples: list[dict] = []
    n_batches = math.ceil(len(dataset) / args.batch_size)
    for b in range(n_batches):
        batch = dataset[b * args.batch_size : (b + 1) * args.batch_size]
        res = pipeline.generate([ex["input"] for ex in batch])
        strings = res["string"]
        if isinstance(strings, str):
            strings = [strings]
        for i, ex in enumerate(batch):
            raw = ex["input"]["raw_output"]
            answers = raw if isinstance(raw, list) else [raw]
            exp_words = [first_word(a) for a in answers]
            gen_word = predict(strings[i], vocab, vocab_da)
            hit = gen_word in exp_words
            hit_da = deaccent(gen_word) in [deaccent(w) for w in exp_words]
            correct += hit
            correct_da += hit_da
            total += 1
            pc = per_class.setdefault(exp_words[0], {"correct": 0, "total": 0})
            pc["correct"] += int(hit)
            pc["total"] += 1
            if len(examples) < 40:
                examples.append(
                    {
                        "prompt": ex["input"].get("raw_input") or "",
                        "expected": exp_words,
                        "generated": strings[i].strip(),
                        "gen_word": gen_word,
                        "hit": bool(hit),
                    }
                )

    acc = correct / total if total else 0.0
    acc_da = correct_da / total if total else 0.0
    result = {
        "task": args.task,
        "model": args.model,
        "scorer": "gen_first_cycle_word_match",
        "max_new_tokens": args.max_new_tokens,
        "accuracy": acc,
        "accuracy_accent_insensitive": acc_da,
        "gate_threshold": args.gate,
        "encoding_gate_pass": acc >= args.gate,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }
    with open(os.path.join(out_dir, "accuracy.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(
        f"[{args.task}] accuracy={acc:.3f} (accent-insensitive {acc_da:.3f}) "
        f"gate@{args.gate}={'PASS' if acc >= args.gate else 'FAIL'}  ({correct}/{total})  -> {out_dir}/accuracy.json"
    )


if __name__ == "__main__":
    main()
