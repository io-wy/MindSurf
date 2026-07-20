"""Run the equal-budget official-vs-team data ablation and cross-evaluation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402
from python_starter.core.dataset_registry import DatasetRegistry  # noqa: E402

ARM_NAMES = ("official", "team")


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def _run_parallel(commands: list[list[str]]) -> None:
    processes = [subprocess.Popen(command, cwd=ROOT) for command in commands]
    failures = [process.wait() for process in processes]
    if any(failures):
        raise subprocess.CalledProcessError(max(failures), "parallel formal pipelines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--parallel-training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluation-memory-mib", type=int, default=4096)
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=ROOT / "configs/datasets/index.json",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Use completed arm checkpoints instead of launching training",
    )
    args = parser.parse_args()

    registry = DatasetRegistry(ROOT, args.dataset_index)
    arms = {name: registry.resolve(name) for name in ARM_NAMES}
    for record in arms.values():
        registry.validate_identity(record)

    suffix = "_80m_smoke" if args.smoke else "_80m"
    if not args.evaluation_only:
        commands = []
        for arm in arms:
            command = [
                sys.executable,
                "scripts/run_formal_pipeline.py",
                "--dataset",
                arm,
                "--dataset-index",
                str(args.dataset_index),
                "--device",
                args.device,
            ]
            if args.smoke:
                command.append("--smoke")
            commands.append(command)
        if args.parallel_training:
            _run_parallel(commands)
        else:
            for command in commands:
                _run(command)

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    training_summaries: dict[str, dict[str, Any]] = {}
    for trained_arm, trained_record in arms.items():
        trained_identity = trained_record.identity
        matrix[trained_arm] = {}
        checkpoint = ROOT / f"models/checkpoints/{trained_identity}{suffix}/final_model.pt"
        training_summary_path = (
            ROOT / "models/checkpoints" / f"{trained_identity}{suffix}" / "training_summary.json"
        )
        training_summaries[trained_arm] = json.loads(
            training_summary_path.read_text(encoding="utf-8")
        )
        for evaluated_arm, evaluated_record in arms.items():
            evaluated_identity = evaluated_record.identity
            output = (
                ROOT
                / "artifacts/evaluation"
                / f"{trained_identity}{suffix}_on_{evaluated_identity}.json"
            )
            evaluation_command = [
                sys.executable,
                "scripts/evaluate_candidate.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(trained_record.tokenizer),
                "--validation",
                str(evaluated_record.validation),
                "--test",
                str(evaluated_record.test),
                "--audit",
                str(evaluated_record.audit),
                "--output",
                str(output),
                "--device",
                args.device,
                *(
                    [
                        "--strict-batches",
                        "5",
                        "--domain-blocks",
                        "5",
                        "--fixed-new-tokens",
                        "16",
                    ]
                    if args.smoke
                    else []
                ),
            ]
            _run(
                [
                    sys.executable,
                    "scripts/run_with_gpu_lease.py",
                    "--run-id",
                    f"cross-eval-{trained_arm}-on-{evaluated_arm}{'-smoke' if args.smoke else ''}",
                    "--required-memory-mib",
                    str(args.evaluation_memory_mib),
                    "--",
                    *evaluation_command,
                ]
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            metrics = result["metrics"]
            matrix[trained_arm][evaluated_arm] = {
                "validation_loss": metrics["strict_val"]["loss"],
                "test_loss": metrics["strict_test"]["loss"],
                "mcq_accuracy": metrics["mcq"]["accuracy"],
                "fixed_prompt_score": metrics["fixed_prompts"]["mean_score"],
                "high_repetition_count": metrics["fixed_prompts"]["repetition_count"],
                "evaluation_sha256": sha256_file(output),
            }

    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "causal_variable": "pretraining_dataset",
        "controlled": {
            "model_config_sha256": sha256_file(ROOT / "configs/model/minimind_80m.yaml"),
            "training_config_sha256": sha256_file(ROOT / "configs/training/pretrain.yaml"),
            "tokenizer_shared": True,
            "seed": 20260511,
            "optimizer_steps": 10000 if not args.smoke else 20,
            "sequence_length": 384,
            "batch_size": 32,
            "seen_tokens": 122880000 if not args.smoke else 245760,
            "evaluation_matrix": "both candidates evaluated on both strict holdouts",
        },
        "dataset_arms": {
            name: {
                "identity": record.identity,
                "dataset_id": record.dataset_id,
                "revision": record.revision,
            }
            for name, record in arms.items()
        },
        "dataset_index_sha256": registry.sha256,
        "training_summaries": training_summaries,
        "matrix": matrix,
        "public_release_performed": False,
    }
    output = ROOT / (
        "artifacts/evaluation/dataset_ablation_smoke.json"
        if args.smoke
        else "artifacts/evaluation/dataset_ablation_80m.json"
    )
    write_json_atomic(output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
