"""Inject a real training-process interruption and verify exact checkpoint recovery."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402


def _torch_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _train_command(
    *,
    output_dir: Path,
    run_name: str,
    max_steps: int,
    save_every: int,
    parent: Path | None = None,
    resume_from: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "data=mindsurf_targeted_stem_v1",
        "seed=31415",
        f"run_name={run_name}",
        f"training.output_dir={output_dir}",
        "training.device=cuda:0",
        f"training.max_steps={max_steps}",
        "training.learning_rate=0.00001",
        "training.warmup_steps=2",
        "training.stable_ratio=0.8",
        f"training.save_every={save_every}",
        "training.checkpoint_keep_last=4",
        "training.eval_every=1000",
        "training.logging_every=10",
    ]
    if parent is not None:
        command.append(f"init_from={parent.resolve()}")
    if resume_from is not None:
        command.append(f"resume_from={resume_from.resolve()}")
    return command


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("models/checkpoints/minimind_official_v1_80m/final_model.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/checkpoints/recovery_drill_seed31415"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/infra/training_recovery_drill.json"),
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--interrupt-step", type=int, default=10)
    args = parser.parse_args()

    baseline_dir = args.output_root / "baseline"
    recovered_dir = args.output_root / "interrupted_then_resumed"
    if baseline_dir.exists() or recovered_dir.exists():
        raise FileExistsError("recovery drill outputs already exist; choose a new output root")

    _run(
        _train_command(
            output_dir=baseline_dir,
            run_name="recovery-drill-baseline-seed31415",
            max_steps=args.max_steps,
            save_every=args.interrupt_step,
            parent=args.parent,
        )
    )

    interrupted = subprocess.Popen(
        _train_command(
            output_dir=recovered_dir,
            run_name="recovery-drill-interrupted-seed31415",
            max_steps=args.max_steps,
            save_every=args.interrupt_step,
            parent=args.parent,
        ),
        cwd=ROOT,
    )
    rolling = recovered_dir / f"checkpoint_step_{args.interrupt_step}.pt"
    deadline = time.monotonic() + 180
    while not rolling.is_file():
        if interrupted.poll() is not None:
            raise RuntimeError(
                f"training exited before the interruption checkpoint: {interrupted.returncode}"
            )
        if time.monotonic() >= deadline:
            interrupted.terminate()
            interrupted.wait(timeout=30)
            raise TimeoutError(f"timed out waiting for {rolling}")
        time.sleep(0.1)
    interrupted.terminate()
    interrupted_return_code = interrupted.wait(timeout=30)

    interruption_checkpoint = _load(rolling)
    interruption_progress = interruption_checkpoint["progress"]
    _run(
        _train_command(
            output_dir=recovered_dir,
            run_name="recovery-drill-resumed-seed31415",
            max_steps=args.max_steps,
            save_every=args.interrupt_step,
            resume_from=rolling,
        )
    )

    baseline_path = baseline_dir / "final_model.pt"
    recovered_path = recovered_dir / "final_model.pt"
    baseline = _load(baseline_path)
    recovered = _load(recovered_path)
    progress_keys = ("global_step", "micro_step", "consumed_blocks", "consumed_tokens")
    progress_continuity = {
        key: baseline["progress"][key] == recovered["progress"][key]
        for key in progress_keys
    }
    model_exact = all(
        torch.equal(value, recovered["model_state_dict"][name])
        for name, value in baseline["model_state_dict"].items()
    )
    checks = {
        "process_was_interrupted": interrupted_return_code != 0,
        "rolling_checkpoint_step_matches": (
            interruption_progress["global_step"] == args.interrupt_step
        ),
        "optimizer_state_exact": (
            _torch_hash(baseline["optimizer_state_dict"])
            == _torch_hash(recovered["optimizer_state_dict"])
        ),
        "scheduler_state_exact": (
            _torch_hash(baseline["scheduler_state_dict"])
            == _torch_hash(recovered["scheduler_state_dict"])
        ),
        "rng_state_exact": (
            _torch_hash(baseline["rng_state"]) == _torch_hash(recovered["rng_state"])
        ),
        "model_state_exact": model_exact,
        **{f"progress_{key}_continuous": value for key, value in progress_continuity.items()},
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "seed": 31415,
        "max_steps": args.max_steps,
        "interrupt_step": args.interrupt_step,
        "interrupted_return_code": interrupted_return_code,
        "checks": checks,
        "interruption_progress": {
            key: interruption_progress[key] for key in progress_keys
        },
        "final_progress": {key: recovered["progress"][key] for key in progress_keys},
        "artifacts": {
            "parent_sha256": sha256_file(args.parent),
            "rolling_checkpoint_sha256": sha256_file(rolling),
            "baseline_final_sha256": sha256_file(baseline_path),
            "recovered_final_sha256": sha256_file(recovered_path),
        },
    }
    write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit("recovery drill did not reproduce the uninterrupted state")


if __name__ == "__main__":
    main()
