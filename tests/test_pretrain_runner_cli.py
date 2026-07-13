from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


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
    assert "--trackio_project" in result.stdout
    assert "--trackio_space_id" in result.stdout


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
    assert summary["status"] == "completed"
    assert Path(summary["metrics_path"]).exists()
    assert Path(summary["run_manifest_path"]).exists()

    metrics_rows = [
        json.loads(line)
        for line in Path(summary["metrics_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in metrics_rows].count("start") == 2
    assert any(row["event"] == "alert" and row["title"] == "training resumed" for row in metrics_rows)
    assert [row["status"] for row in metrics_rows if row["event"] == "finish"] == ["stopped", "completed"]


def test_resume_is_exact_with_accumulation_and_initial_skip(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "".join(
            json.dumps({"text": f"deterministic row {index} abcdefghijklmnopqrstuvwxyz"}) + "\n"
            for index in range(32)
        ),
        encoding="utf-8",
    )

    def command(output_dir: Path, save_weight: str) -> list[str]:
        return [
            sys.executable,
            "experiments/pretrain/scripts/train_pretrain_optimized.py",
            "--data_path", str(data_path),
            "--tokenizer_path", "model",
            "--save_dir", str(output_dir),
            "--save_weight", save_weight,
            "--stream_packed",
            "--skip_blocks", "3",
            "--max_steps", "4",
            "--batch_size", "1",
            "--accumulation_steps", "2",
            "--max_seq_len", "8",
            "--hidden_size", "32",
            "--num_hidden_layers", "1",
            "--num_attention_heads", "4",
            "--num_key_value_heads", "2",
            "--intermediate_size", "64",
            "--device", "cpu",
            "--checkpoint_interval", "1",
            "--log_interval", "1",
        ]

    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    subprocess.run(command(uninterrupted_dir, "full"), check=True)
    subprocess.run(command(resumed_dir, "resume") + ["--stop_after_steps", "1"], check=True)
    stopped = json.loads((resumed_dir / "resume_32_summary.json").read_text(encoding="utf-8"))
    checkpoint = resumed_dir / "resume_training_state.pt"
    subprocess.run(command(resumed_dir, "resume") + ["--resume_checkpoint", str(checkpoint)], check=True)

    completed = json.loads((resumed_dir / "resume_32_summary.json").read_text(encoding="utf-8"))
    full_weights = torch.load(uninterrupted_dir / "full_32.pth", map_location="cpu", weights_only=True)
    resumed_weights = torch.load(resumed_dir / "resume_32.pth", map_location="cpu", weights_only=True)

    assert stopped["steps"] == 2
    assert stopped["consumed_blocks"] == 5
    assert completed["consumed_blocks"] == 7
    assert completed["optimizer_steps"] == 2
    assert all(torch.equal(full_weights[name], resumed_weights[name]) for name in full_weights)


def test_exhausted_stream_replays_uncommitted_microbatch_on_resume(tmp_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained("model")
    max_seq_len = 8
    first_text = None
    first_blocks = None
    for repeats in range(4, 200):
        candidate = "alpha " * repeats
        blocks = (len(tokenizer(candidate, add_special_tokens=False).input_ids) + 1) // max_seq_len
        if blocks >= 3 and blocks % 2 == 1:
            first_text = candidate
            first_blocks = blocks
            break
    assert first_text is not None and first_blocks is not None
    second_text = "beta " * 40
    total_steps = first_blocks + 1
    interrupted_data = tmp_path / "interrupted.jsonl"
    full_data = tmp_path / "full.jsonl"
    first_row = json.dumps({"text": first_text}) + "\n"
    second_row = json.dumps({"text": second_text}) + "\n"
    interrupted_data.write_text(first_row, encoding="utf-8")
    full_data.write_text(first_row + second_row, encoding="utf-8")

    def command(data_path: Path, output_dir: Path, name: str) -> list[str]:
        return [
            sys.executable,
            "experiments/pretrain/scripts/train_pretrain_optimized.py",
            "--data_path", str(data_path),
            "--tokenizer_path", "model",
            "--save_dir", str(output_dir),
            "--save_weight", name,
            "--stream_packed",
            "--max_steps", str(total_steps),
            "--batch_size", "1",
            "--accumulation_steps", "2",
            "--max_seq_len", str(max_seq_len),
            "--hidden_size", "32",
            "--num_hidden_layers", "1",
            "--num_attention_heads", "4",
            "--num_key_value_heads", "2",
            "--intermediate_size", "64",
            "--device", "cpu",
            "--checkpoint_interval", "1",
            "--log_interval", "100",
        ]

    full_dir = tmp_path / "full-run"
    resumed_dir = tmp_path / "resumed-run"
    subprocess.run(command(full_data, full_dir, "full"), check=True)
    subprocess.run(command(interrupted_data, resumed_dir, "resume"), check=True)
    exhausted = json.loads((resumed_dir / "resume_32_summary.json").read_text(encoding="utf-8"))
    checkpoint = resumed_dir / "resume_training_state.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert exhausted["steps"] == first_blocks
    assert exhausted["checkpoint_deferred"] is True
    assert saved["progress"]["global_step"] == first_blocks - 1
    assert saved["progress"]["consumed_blocks"] == first_blocks - 1

    interrupted_data.write_text(first_row + second_row, encoding="utf-8")
    subprocess.run(
        command(interrupted_data, resumed_dir, "resume") + ["--resume_checkpoint", str(checkpoint)],
        check=True,
    )
    full_weights = torch.load(full_dir / "full_32.pth", map_location="cpu", weights_only=True)
    resumed_weights = torch.load(resumed_dir / "resume_32.pth", map_location="cpu", weights_only=True)
    assert all(torch.equal(full_weights[name], resumed_weights[name]) for name in full_weights)
