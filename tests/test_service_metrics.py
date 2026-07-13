from scripts import serve_openai_api as service


def test_native_service_exposes_dependency_free_prometheus_metrics() -> None:
    with service.request_stats_lock:
        service.request_stats.update(
            {
                "requests": 7,
                "errors": 1,
                "duration_seconds_sum": 2.5,
            }
        )
    with service.batch_stats_lock:
        service.batch_stats.update({"batches": 3, "jobs": 6, "max_batch_size": 4})

    output = service.render_prometheus_metrics()

    assert "minimind_requests_total 7" in output
    assert "minimind_request_errors_total 1" in output
    assert "minimind_request_duration_seconds_sum 2.5" in output
    assert "minimind_batches_total 3" in output
    assert "minimind_batch_jobs_total 6" in output
    assert "minimind_batch_max_size 4" in output
