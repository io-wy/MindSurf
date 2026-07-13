from pathlib import Path


def test_infra_dependencies_and_bootstrap_are_pinned() -> None:
    extras = Path("experiments/pretrain/requirements-infra.txt").read_text(encoding="utf-8")
    bootstrap = Path("experiments/pretrain/scripts/bootstrap_vllm_env.sh").read_text(encoding="utf-8")

    assert "trackio==0.30.3" in extras
    assert "transformers==5.13.1" in extras
    assert "huggingface-hub==1.23.0" in extras
    assert "swanlab==0.8.4" in extras
    assert "wandb==0.28.0" in extras
    assert 'VLLM_VERSION="${VLLM_VERSION:-0.25.0}"' in bootstrap
    assert '"$VLLM_ENV/bin/pip" check' in bootstrap
    assert "uninstall -y torchcodec" not in bootstrap
    assert "PyNvVideoCodec" in bootstrap
    assert "/home/" not in bootstrap


def test_main_requirements_use_the_compatible_hf_observability_stack() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    for pin in (
        "datasets==5.0.0",
        "huggingface-hub==1.23.0",
        "sentence_transformers==5.6.0",
        "swanlab==0.8.4",
        "trackio==0.30.3",
        "transformers==5.13.1",
        "trl==1.8.0",
        "wandb==0.28.0",
    ):
        assert pin in requirements


def test_ci_is_read_only_and_runs_infra_verification() -> None:
    workflow = Path(".github/workflows/infra-ci.yml").read_text(encoding="utf-8")

    assert "contents: read" in workflow
    assert "python -m pytest -q" in workflow
    assert "build_asset_manifest.py" in workflow
    assert "git diff --check" in workflow
    assert r"\.(pth|pt|ckpt|safetensors)$" in workflow


def test_operations_doc_uses_portable_paths_and_real_commands() -> None:
    document = Path("docs/infra/single_gpu_operations.md").read_text(encoding="utf-8")

    assert "release_gate.py" in document
    assert "start_vllm_tmux.sh" in document
    assert "--trackio_project" in document
    assert "D:\\" not in document
    assert "/home/" not in document
    assert "192.168." not in document
