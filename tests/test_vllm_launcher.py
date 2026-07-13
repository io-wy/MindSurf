from pathlib import Path


def test_vllm_launcher_uses_small_model_defaults() -> None:
    launcher = Path("experiments/pretrain/scripts/serve_vllm.sh")
    text = launcher.read_text(encoding="utf-8")

    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"' in text
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"' in text
    assert 'VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"' in text
    assert "--model-impl transformers" in text
    assert '"$@"' in text
