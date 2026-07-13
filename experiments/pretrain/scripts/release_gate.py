from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CHUNK_SIZE = 1024 * 1024


def _resolve(root: Path, raw_path: str) -> Path:
    root = root.resolve()
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"release path escapes root: {raw_path}")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _check_threshold(name: str, value: Any, rule: dict[str, Any], failures: list[str]) -> None:
    if "equals" in rule and value != rule["equals"]:
        failures.append(f"{name} expected {rule['equals']!r}, got {value!r}")
    if "min" in rule and (not isinstance(value, (int, float)) or value < rule["min"]):
        failures.append(f"{name} expected >= {rule['min']}, got {value!r}")
    if "max" in rule and (not isinstance(value, (int, float)) or value > rule["max"]):
        failures.append(f"{name} expected <= {rule['max']}, got {value!r}")


def _find_case(rows: list[dict[str, Any]], prompt_chars: int, concurrency: int) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("prompt_chars") == prompt_chars and row.get("concurrency") == concurrency
        ),
        None,
    )


def evaluate_release(spec_path: Path, root: Path, *, check_service: bool = False) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    spec = _read_json(spec_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    if spec.get("schema_version") != 1:
        failures.append(f"unsupported release schema: {spec.get('schema_version')!r}")

    assets = spec["assets"]
    for name in ("checkpoint", "model"):
        definition = assets[name]
        path = _resolve(root, definition["path"])
        if not path.is_file():
            failures.append(f"{name} missing: {definition['path']}")
            continue
        actual_size = path.stat().st_size
        actual_sha = _sha256(path)
        checks[name] = {"size_bytes": actual_size, "sha256": actual_sha}
        if actual_size != definition["size_bytes"]:
            failures.append(f"{name} size expected {definition['size_bytes']}, got {actual_size}")
        if actual_sha != definition["sha256"]:
            failures.append(f"{name} sha256 expected {definition['sha256']}, got {actual_sha}")

    manifest_path = _resolve(root, assets["artifact_manifest"])
    if not manifest_path.is_file():
        failures.append(f"artifact manifest missing: {assets['artifact_manifest']}")
    else:
        artifact_manifest = _read_json(manifest_path)
        checkpoint_sha = assets["checkpoint"]["sha256"]
        if artifact_manifest.get("source_weight", {}).get("sha256") != checkpoint_sha:
            failures.append("artifact manifest source_weight sha256 does not match checkpoint")
        model_entry = next(
            (item for item in artifact_manifest.get("files", []) if item.get("name") == "model.safetensors"),
            None,
        )
        if model_entry is None or model_entry.get("sha256") != assets["model"]["sha256"]:
            failures.append("artifact manifest model.safetensors sha256 does not match release spec")

    evaluation_checks = {}
    for name, rule in spec.get("evaluation", {}).items():
        path = _resolve(root, rule["path"])
        if not path.is_file():
            failures.append(f"evaluation {name} missing: {rule['path']}")
            continue
        payload = _read_json(path)
        value = payload.get(rule["field"])
        evaluation_checks[name] = value
        _check_threshold(name, value, rule, failures)
    checks["evaluation"] = evaluation_checks

    serving = spec.get("serving", {})
    candidate_path = _resolve(root, serving["candidate_summary"])
    baseline_path = _resolve(root, serving["baseline_summary"])
    if not candidate_path.is_file() or not baseline_path.is_file():
        failures.append("serving benchmark summary is missing")
    else:
        candidate_rows = _read_json(candidate_path)
        baseline_rows = _read_json(baseline_path)
        total_requests = sum(int(row.get("requests", 0)) for row in candidate_rows)
        total_errors = sum(int(row.get("errors", 0)) for row in candidate_rows)
        checks["serving"] = {"total_requests": total_requests, "total_errors": total_errors, "cases": []}
        if total_requests < serving.get("minimum_total_requests", 0):
            failures.append(
                f"serving requests expected >= {serving['minimum_total_requests']}, got {total_requests}"
            )
        if total_errors > serving.get("maximum_total_errors", 0):
            failures.append(f"serving errors expected <= {serving['maximum_total_errors']}, got {total_errors}")
        for comparison in serving.get("comparisons", []):
            prompt_chars = comparison["prompt_chars"]
            concurrency = comparison["concurrency"]
            candidate = _find_case(candidate_rows, prompt_chars, concurrency)
            baseline = _find_case(baseline_rows, prompt_chars, concurrency)
            if candidate is None or baseline is None:
                failures.append(f"serving comparison missing prompt={prompt_chars} concurrency={concurrency}")
                continue
            candidate_ms = candidate.get("median_total_ms")
            baseline_ms = baseline.get("median_total_ms")
            ratio = candidate_ms / baseline_ms if baseline_ms else float("inf")
            checks["serving"]["cases"].append(
                {
                    "prompt_chars": prompt_chars,
                    "concurrency": concurrency,
                    "candidate_ms": candidate_ms,
                    "baseline_ms": baseline_ms,
                    "ratio": ratio,
                }
            )
            if candidate_ms > comparison["max_candidate_ms"]:
                failures.append(
                    f"candidate latency prompt={prompt_chars} concurrency={concurrency} "
                    f"expected <= {comparison['max_candidate_ms']}, got {candidate_ms}"
                )
            if ratio > comparison["max_ratio"]:
                failures.append(
                    f"candidate ratio prompt={prompt_chars} concurrency={concurrency} "
                    f"expected <= {comparison['max_ratio']}, got {ratio:.4f}"
                )

    if check_service:
        service_checks = []
        for endpoint in serving.get("endpoints", []):
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    status = response.status
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                failures.append(f"service endpoint failed {endpoint}: {exc}")
                status = None
            if status != 200:
                failures.append(f"service endpoint expected 200 {endpoint}, got {status}")
            service_checks.append({"endpoint": endpoint, "status": status})
        checks["service_endpoints"] = service_checks

    return {
        "schema_version": 1,
        "release_id": spec.get("release_id"),
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a MiniMind release before promotion")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-service", action="store_true")
    args = parser.parse_args()

    report = evaluate_release(args.spec, args.root, check_service=args.check_service)
    _atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
