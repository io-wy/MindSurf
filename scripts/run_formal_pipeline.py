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


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=("official", "team"),
        default="official",
        help="Run one frozen data arm; use run_dataset_ablation.py for both",
    )
    args = parser.parse_args()

    datasets = {
        "official": {
            "hydra": "minimind_official_v1",
            "identity": "minimind_official_v1",
            "run_name": "minimind-official-v1-80m-wsd",
        },
        "team": {
            "hydra": "mindsurf_team_v1",
            "identity": "mindsurf_team_v1",
            "run_name": "mindsurf-team-v1-80m-wsd",
        },
    }
    selected = datasets[args.dataset]
    identity = selected["identity"]
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
            f"artifacts/data/{identity}/audit.json",
            "--spec",
            f"configs/datasets/{identity}.json",
            "--training-view-manifest",
            f"artifacts/data/{identity}/training_view.json",
            "--train-path",
            f"data/processed/{identity}/pretrain_train_nfkc_dedup.jsonl",
            "--require-cuda",
            "--output-dir",
            str(checkpoint_dir),
        ]
    )
    train_command = [
        sys.executable,
        "scripts/train.py",
        f"data={selected['hydra']}",
        f"run_name={selected['run_name']}",
        f"training.output_dir=models/checkpoints/{identity}_80m",
        f"training.device={args.device}",
    ]
    if args.smoke:
        train_command.extend(
            [
                f"run_name={selected['run_name']}-smoke",
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
    _run(train_command)
    if args.smoke:
        checkpoint = ROOT / f"models/checkpoints/{identity}_80m_smoke/final_model.pt"
        training_summary = ROOT / f"models/checkpoints/{identity}_80m_smoke/training_summary.json"
        evaluation = ROOT / f"artifacts/evaluation/{identity}_80m_smoke.json"

    _run(
        [
            sys.executable,
            "scripts/evaluate_candidate.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            f"data/raw/{identity}/tokenizer",
            "--validation",
            f"data/raw/{identity}/strict_splits/pretrain_strict_val_2k.jsonl",
            "--test",
            f"data/raw/{identity}/strict_splits/pretrain_strict_test_2k.jsonl",
            "--audit",
            f"artifacts/data/{identity}/audit.json",
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
            ]
        )
        registered = True

    write_json_atomic(
        outcome_path,
        {
            "schema_version": 1,
            "completed_at": datetime.now(UTC).isoformat(),
            "smoke": args.smoke,
            "dataset_arm": args.dataset,
            "dataset_identity": identity,
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
