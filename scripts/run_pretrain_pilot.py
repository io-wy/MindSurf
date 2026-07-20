"""Run and evaluate the targeted STEM continuation pilot from the Official parent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402
from python_starter.core.dataset_registry import DatasetRegistry  # noqa: E402
from python_starter.core.evaluation import compare_mcq_items  # noqa: E402


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def _strict_mean(result: dict[str, Any]) -> float:
    return (
        float(result["metrics"]["strict_val"]["loss"])
        + float(result["metrics"]["strict_test"]["loss"])
    ) / 2


def _category_accuracy(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value["accuracy"])
        for name, value in result["metrics"]["mcq"]["by_category"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("models/checkpoints/minimind_official_v1_80m/final_model.pt"),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/experiments/targeted_stem_pilot_80m.json"),
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=Path("configs/datasets/index.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints/targeted_stem_pilot_seed42"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/experiments/targeted_stem_pilot_seed42"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-memory-mib", type=int, default=11_000)
    parser.add_argument("--evaluation-memory-mib", type=int, default=4096)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pilot = spec["pilot"]
    registry = DatasetRegistry(ROOT, args.dataset_index)
    continuation = registry.resolve(spec["continuation_dataset"])
    evaluation_sets = {
        "official": registry.resolve("official"),
        "team": registry.resolve("team"),
    }
    registry.validate_identity(continuation)
    for record in evaluation_sets.values():
        registry.validate_identity(record)

    _run(
        [
            sys.executable,
            "scripts/preflight_training.py",
            "--audit",
            str(continuation.audit),
            "--spec",
            str(continuation.spec),
            "--training-view-manifest",
            str(continuation.training_view_manifest),
            "--train-path",
            str(continuation.train),
            "--require-cuda",
            "--required-gpu-memory-mib",
            str(args.training_memory_mib),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    train_command = [
        sys.executable,
        "scripts/train.py",
        f"data={continuation.hydra_config}",
        f"seed={pilot['seed']}",
        f"init_from={args.parent.resolve()}",
        "run_name=targeted-stem-pilot-seed42",
        f"training.output_dir={args.output_dir}",
        f"training.device={args.device}",
        f"training.max_steps={pilot['steps']}",
        f"training.learning_rate={pilot['learning_rate']}",
        f"training.warmup_steps={pilot['warmup_steps']}",
        "training.stable_ratio=0.8",
        "training.save_every=250",
        "training.eval_every=500",
        "training.logging_every=10",
    ]
    _run(
        [
            sys.executable,
            "scripts/run_with_gpu_lease.py",
            "--run-id",
            "targeted-stem-pilot-seed42",
            "--required-memory-mib",
            str(args.training_memory_mib),
            "--",
            *train_command,
        ]
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = args.output_dir / "final_model.pt"
    results: dict[str, dict[str, dict[str, Any]]] = {"parent": {}, "candidate": {}}
    for candidate_name, checkpoint in (
        ("parent", args.parent),
        ("candidate", candidate_checkpoint),
    ):
        for dataset_name, dataset in evaluation_sets.items():
            output = args.artifact_dir / f"{candidate_name}_on_{dataset_name}.json"
            evaluation_command = [
                sys.executable,
                "scripts/evaluate_candidate.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(evaluation_sets["official"].tokenizer),
                "--validation",
                str(dataset.validation),
                "--test",
                str(dataset.test),
                "--audit",
                str(dataset.audit),
                "--output",
                str(output),
                "--device",
                args.device,
            ]
            _run(
                [
                    sys.executable,
                    "scripts/run_with_gpu_lease.py",
                    "--run-id",
                    f"targeted-stem-{candidate_name}-on-{dataset_name}",
                    "--required-memory-mib",
                    str(args.evaluation_memory_mib),
                    "--",
                    *evaluation_command,
                ]
            )
            results[candidate_name][dataset_name] = json.loads(output.read_text(encoding="utf-8"))

    official_parent = results["parent"]["official"]
    official_candidate = results["candidate"]["official"]
    comparison = compare_mcq_items(
        official_parent["metrics"]["mcq"]["items"],
        official_candidate["metrics"]["mcq"]["items"],
    )
    parent_categories = _category_accuracy(official_parent)
    candidate_categories = _category_accuracy(official_candidate)
    category_deltas = {
        name: candidate_categories[name] - parent_categories[name]
        for name in sorted(parent_categories)
    }
    triggers = spec["triggers"]
    checks = {
        "official_strict_regression": (
            _strict_mean(official_candidate) - _strict_mean(official_parent)
            <= float(triggers["official_strict_mean_regression_max"])
        ),
        "mcq_accuracy_gain": (
            float(comparison["accuracy_delta"]) >= float(triggers["mcq_accuracy_delta_min"])
        ),
        "mcq_category_regression": (
            min(category_deltas.values()) >= -float(triggers["mcq_regressed_category_max"])
        ),
        "generation_repetition": (
            int(official_candidate["metrics"]["generation"]["repetition_count"])
            - int(official_parent["metrics"]["generation"]["repetition_count"])
            <= int(triggers["generation_repetition_increase_max"])
        ),
    }
    outcome = {
        "schema_version": 1,
        "hypothesis": spec["hypothesis"],
        "experiment_spec_sha256": sha256_file(args.spec),
        "dataset_index_sha256": registry.sha256,
        "parent_checkpoint_sha256": sha256_file(args.parent),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "pilot": pilot,
        "strict_mean": {
            candidate_name: {
                dataset_name: _strict_mean(value) for dataset_name, value in by_dataset.items()
            }
            for candidate_name, by_dataset in results.items()
        },
        "mcq_comparison": comparison,
        "mcq_category_deltas": category_deltas,
        "checks": checks,
        "continue_to_formal": all(checks.values()),
        "artifacts": {
            candidate_name: {
                dataset_name: sha256_file(
                    args.artifact_dir / f"{candidate_name}_on_{dataset_name}.json"
                )
                for dataset_name in by_dataset
            }
            for candidate_name, by_dataset in results.items()
        },
    }
    write_json_atomic(args.artifact_dir / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
