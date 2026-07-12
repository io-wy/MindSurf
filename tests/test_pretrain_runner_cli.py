from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


def test_pretrain_runner_exposes_periodic_resume_checkpoint_flags() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/pretrain/scripts/train_pretrain_optimized.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--resume_checkpoint" in result.stdout
    assert "--checkpoint_interval" in result.stdout


def test_pretrain_runner_resumes_tiny_cpu_run(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "".join(
            json.dumps({"text": f"tiny training row {index} abcdefghijklmnopqrstuvwxyz"}) + "\n"
            for index in range(16)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    checkpoint_path = output_dir / "tiny_training_state.pt"
    base_command = [
        sys.executable,
        "experiments/pretrain/scripts/train_pretrain_optimized.py",
        "--data_path",
        str(data_path),
        "--tokenizer_path",
        "model",
        "--save_dir",
        str(output_dir),
        "--save_weight",
        "tiny",
        "--stream_packed",
        "--max_steps",
        "2",
        "--batch_size",
        "2",
        "--max_seq_len",
        "8",
        "--hidden_size",
        "32",
        "--num_hidden_layers",
        "1",
        "--num_attention_heads",
        "4",
        "--num_key_value_heads",
        "2",
        "--intermediate_size",
        "64",
        "--device",
        "cpu",
        "--checkpoint_interval",
        "1",
        "--log_interval",
        "1",
    ]

    subprocess.run(base_command + ["--stop_after_steps", "1"], check=True)
    assert checkpoint_path.exists()

    subprocess.run(
        base_command + ["--resume_checkpoint", str(checkpoint_path)],
        check=True,
    )

    summary = json.loads((output_dir / "tiny_32_summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 2
    assert summary["optimizer_steps"] == 2
    assert summary["consumed_blocks"] == 4
    assert summary["resumed_from"] == str(checkpoint_path)
