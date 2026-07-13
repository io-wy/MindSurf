from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiments.pretrain.scripts.promote_release import promote_release
from experiments.pretrain.scripts.release_gate import evaluate_release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release_fixture(root: Path) -> Path:
    checkpoint = root / "runs/model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    model_file = root / "out/artifact/model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"safetensors")
    _write_json(
        root / "out/artifact/artifact_manifest.json",
        {
            "source_weight": {"sha256": _sha256(checkpoint)},
            "files": [{"name": "model.safetensors", "sha256": _sha256(model_file)}],
        },
    )
    _write_json(root / "eval/val.json", {"mean_loss": 2.5})
    _write_json(root / "eval/test.json", {"mean_loss": 2.6})
    _write_json(root / "eval/mcq.json", {"accuracy": 0.35})
    _write_json(root / "eval/prompts.json", {"mean_score": 0.3, "empty_count": 4})
    _write_json(root / "eval/validation.json", {"max_abs_diff": 0.005, "generation_all_exact": True})
    _write_json(
        root / "bench/native.json",
        [{"prompt_chars": 128, "concurrency": 16, "requests": 256, "errors": 0, "median_total_ms": 1000.0}],
    )
    _write_json(
        root / "bench/vllm.json",
        [{"prompt_chars": 128, "concurrency": 16, "requests": 256, "errors": 0, "median_total_ms": 280.0}],
    )
    spec_path = root / "release.json"
    _write_json(
        spec_path,
        {
            "schema_version": 1,
            "release_id": "tiny-release",
            "artifact_dir": "out/artifact",
            "assets": {
                "checkpoint": {"path": "runs/model.pth", "sha256": _sha256(checkpoint), "size_bytes": 10},
                "model": {"path": "out/artifact/model.safetensors", "sha256": _sha256(model_file), "size_bytes": 11},
                "artifact_manifest": "out/artifact/artifact_manifest.json",
            },
            "evaluation": {
                "strict_val": {"path": "eval/val.json", "field": "mean_loss", "max": 3.0},
                "strict_test": {"path": "eval/test.json", "field": "mean_loss", "max": 3.0},
                "mcq": {"path": "eval/mcq.json", "field": "accuracy", "min": 0.3},
                "fixed_prompt_score": {"path": "eval/prompts.json", "field": "mean_score", "min": 0.25},
                "fixed_prompt_empty": {"path": "eval/prompts.json", "field": "empty_count", "max": 5},
                "export_diff": {"path": "eval/validation.json", "field": "max_abs_diff", "max": 0.01},
                "generation_exact": {"path": "eval/validation.json", "field": "generation_all_exact", "equals": True},
            },
            "serving": {
                "candidate_summary": "bench/vllm.json",
                "baseline_summary": "bench/native.json",
                "minimum_total_requests": 256,
                "maximum_total_errors": 0,
                "comparisons": [
                    {"prompt_chars": 128, "concurrency": 16, "max_candidate_ms": 350.0, "max_ratio": 0.5}
                ],
            },
        },
    )
    return spec_path


def test_release_gate_checks_hashes_metrics_and_serving(tmp_path: Path) -> None:
    spec_path = _release_fixture(tmp_path)

    report = evaluate_release(spec_path, tmp_path)
    assert report["passed"] is True
    assert not report["failures"]

    (tmp_path / "runs/model.pth").write_bytes(b"corrupt")
    failed = evaluate_release(spec_path, tmp_path)
    assert failed["passed"] is False
    assert any("checkpoint sha256" in failure for failure in failed["failures"])


def test_promotion_writes_atomic_current_pointer_only_after_gate_passes(tmp_path: Path) -> None:
    spec_path = _release_fixture(tmp_path)
    pointer = tmp_path / "out/releases/current.json"

    payload = promote_release(spec_path, tmp_path, pointer)

    assert json.loads(pointer.read_text(encoding="utf-8")) == payload
    assert payload["release_id"] == "tiny-release"
    assert payload["artifact_dir"] == "out/artifact"
    assert not list(pointer.parent.glob("*.tmp"))


def test_promotion_cli_is_runnable_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "experiments/pretrain/scripts/promote_release.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--spec" in result.stdout
