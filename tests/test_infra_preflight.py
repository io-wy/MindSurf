from experiments.pretrain.scripts.infra_preflight import evaluate_snapshot


def test_training_preflight_blocks_busy_gpu_and_low_disk() -> None:
    report = evaluate_snapshot(
        {
            "disk_free_gb": 10.0,
            "gpu_memory_used_mb": 3000,
            "gpu_temperature_c": 70,
            "gpu_compute_processes": 1,
        },
        mode="training",
        min_free_disk_gb=20.0,
        max_gpu_memory_used_mb=1024,
        max_gpu_temperature_c=85,
    )

    assert report["passed"] is False
    assert any("disk" in failure for failure in report["failures"])
    assert any("GPU memory" in failure for failure in report["failures"])
    assert any("compute processes" in failure for failure in report["failures"])


def test_serving_preflight_accepts_small_model_budget() -> None:
    report = evaluate_snapshot(
        {
            "disk_free_gb": 100.0,
            "gpu_memory_used_mb": 0,
            "gpu_temperature_c": 35,
            "gpu_compute_processes": 0,
        },
        mode="serving",
        min_free_disk_gb=10.0,
        max_gpu_memory_used_mb=512,
        max_gpu_temperature_c=85,
    )

    assert report["passed"] is True
    assert not report["failures"]
