from pathlib import Path


def test_vllm_launcher_uses_small_model_defaults() -> None:
    launcher = Path("experiments/pretrain/scripts/serve_vllm.sh")
    text = launcher.read_text(encoding="utf-8")

    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"' in text
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"' in text
    assert 'VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"' in text
    assert "--model-impl transformers" in text
    assert '"$@"' in text


def test_vllm_tmux_lifecycle_scripts_share_one_session_name() -> None:
    scripts = [
        Path("experiments/pretrain/scripts/start_vllm_tmux.sh"),
        Path("experiments/pretrain/scripts/status_vllm_tmux.sh"),
        Path("experiments/pretrain/scripts/stop_vllm_tmux.sh"),
    ]
    contents = [path.read_text(encoding="utf-8") for path in scripts]

    assert all('SESSION="${SESSION:-minimind_vllm_service}"' in text for text in contents)
    assert "infra_preflight.py" in contents[0]
    assert "/health" in contents[0]
    assert "/metrics" in contents[1]
