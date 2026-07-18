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

ARMS = {
    "official": "minimind_official_v1",
    "team": "mindsurf_team_v1",
}


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Use completed arm checkpoints instead of launching training",
    )
    args = parser.parse_args()

    suffix = "_80m_smoke" if args.smoke else "_80m"
    if not args.evaluation_only:
        for arm in ARMS:
            command = [
                sys.executable,
                "scripts/run_formal_pipeline.py",
                "--dataset",
                arm,
                "--device",
                args.device,
            ]
            if args.smoke:
                command.append("--smoke")
            _run(command)

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    training_summaries: dict[str, dict[str, Any]] = {}
    for trained_arm, trained_identity in ARMS.items():
        matrix[trained_arm] = {}
        checkpoint = ROOT / f"models/checkpoints/{trained_identity}{suffix}/final_model.pt"
        training_summary_path = (
            ROOT / "models/checkpoints" / f"{trained_identity}{suffix}" / "training_summary.json"
        )
        training_summaries[trained_arm] = json.loads(
            training_summary_path.read_text(encoding="utf-8")
        )
        for evaluated_arm, evaluated_identity in ARMS.items():
            output = (
                ROOT
                / "artifacts/evaluation"
                / f"{trained_identity}{suffix}_on_{evaluated_identity}.json"
            )
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_candidate.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--tokenizer",
                    f"data/raw/{trained_identity}/tokenizer",
                    "--validation",
                    f"data/raw/{evaluated_identity}/strict_splits/pretrain_strict_val_2k.jsonl",
                    "--test",
                    f"data/raw/{evaluated_identity}/strict_splits/pretrain_strict_test_2k.jsonl",
                    "--audit",
                    f"artifacts/data/{evaluated_identity}/audit.json",
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
        "dataset_arms": ARMS,
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
