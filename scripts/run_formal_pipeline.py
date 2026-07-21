"""Run preflight, training, evaluation, and conditional candidate registration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.core.dataset_registry import DatasetRegistry  # noqa: E402


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--training-memory-mib", type=int, default=11_000)
    parser.add_argument("--evaluation-memory-mib", type=int, default=4096)
    parser.add_argument(
        "--dataset",
        default="official",
        help="Dataset registry name or alias; use run_dataset_ablation.py for both baselines",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=ROOT / "configs/datasets/index.json",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Distinguish an arm from the frozen baseline run of the same dataset",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override appended to the training command; repeatable",
    )
    args = parser.parse_args()

    registry = DatasetRegistry(ROOT, args.dataset_index)
    selected = registry.resolve(args.dataset)
    registry.validate_identity(selected)
    identity = f"{selected.identity}{args.run_suffix}"
    checkpoint_dir = ROOT / f"models/checkpoints/{identity}_80m"
    checkpoint = checkpoint_dir / "final_model.pt"
    training_summary = checkpoint_dir / "training_summary.json"
    evaluation = ROOT / f"artifacts/evaluation/{identity}_80m.json"
    outcome_path = ROOT / f"artifacts/evaluation/{identity}_formal_pipeline_outcome.json"

    _run(
        [
            sys.executable,
            "scripts/preflight_training.py",
            "--audit",
            str(selected.audit),
            "--spec",
            str(selected.spec),
            "--training-view-manifest",
            str(selected.training_view_manifest),
            "--train-path",
            str(selected.train),
            "--require-cuda",
            "--required-gpu-memory-mib",
            str(args.training_memory_mib),
            "--output-dir",
            str(checkpoint_dir),
        ]
    )
    train_command = [
        sys.executable,
        "scripts/train.py",
        f"data={selected.hydra_config}",
        f"run_name={identity}-80m-wsd",
        f"training.output_dir=models/checkpoints/{identity}_80m",
        f"training.device={args.device}",
    ]
    if args.smoke:
        train_command.extend(
            [
                f"run_name={identity}-80m-wsd-smoke",
                "training.max_steps=20",
                "training.warmup_steps=2",
                "training.eval_every=20",
                "training.save_every=10",
                "training.eval_batches=5",
                f"training.output_dir=models/checkpoints/{identity}_80m_smoke",
            ]
        )
    if args.resume:
        train_command.append(f"resume_from={args.resume.resolve()}")
    train_command.extend(args.override)
    _run(
        [
            sys.executable,
            "scripts/run_with_gpu_lease.py",
            "--run-id",
            f"train-{identity}{'-smoke' if args.smoke else ''}",
            "--required-memory-mib",
            str(args.training_memory_mib),
            "--",
            *train_command,
        ]
    )
    if args.smoke:
        checkpoint = ROOT / f"models/checkpoints/{identity}_80m_smoke/final_model.pt"
        training_summary = ROOT / f"models/checkpoints/{identity}_80m_smoke/training_summary.json"
        evaluation = ROOT / f"artifacts/evaluation/{identity}_80m_smoke.json"

    evaluation_command = [
        sys.executable,
        "scripts/evaluate_candidate.py",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(selected.tokenizer),
        "--validation",
        str(selected.validation),
        "--test",
        str(selected.test),
        "--audit",
        str(selected.audit),
        "--output",
        str(evaluation),
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
            f"evaluate-{identity}{'-smoke' if args.smoke else ''}",
            "--required-memory-mib",
            str(args.evaluation_memory_mib),
            "--",
            *evaluation_command,
        ]
    )
    evaluation_data = json.loads(evaluation.read_text(encoding="utf-8"))
    gate = evaluation_data["gate"]
    registered = False
    if gate.get("internal_candidate_passed") is True and not args.smoke:
        _run(
            [
                sys.executable,
                "scripts/register_candidate.py",
                "--name",
                f"{identity}-80m",
                "--checkpoint",
                str(checkpoint),
                "--evaluation",
                str(evaluation),
                "--training-summary",
                str(training_summary),
                "--preflight",
                str(checkpoint_dir / "preflight.json"),
            ]
        )
        registered = True

    write_json_atomic(
        outcome_path,
        {
            "schema_version": 1,
            "completed_at": datetime.now(UTC).isoformat(),
            "smoke": args.smoke,
            "dataset_name": selected.name,
            "dataset_id": selected.dataset_id,
            "dataset_revision": selected.revision,
            "dataset_index_sha256": registry.sha256,
            "checkpoint": str(checkpoint),
            "training_summary": str(training_summary),
            "evaluation": str(evaluation),
            "gate": gate,
            "registered_internal_candidate": registered,
            "public_release_performed": False,
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
