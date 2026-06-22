import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def make_prompt(target_chars: int) -> str:
    prefix = (
        "You are benchmarking a local MiniMind OpenAI-compatible API. "
        "Answer in concise Chinese and keep the response stable. Context: "
    )
    filler = "MiniMind pretrain engineering latency benchmark context fragment. "
    if target_chars <= len(prefix):
        return prefix[:target_chars]
    repeats = ((target_chars - len(prefix)) // len(filler)) + 1
    return (prefix + filler * repeats)[:target_chars]


def post_json(endpoint: str, payload: dict, timeout: float) -> urllib.response.addinfourl:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def run_one(endpoint: str, payload: dict, timeout: float) -> dict:
    start = time.perf_counter()
    first_chunk = None
    output_chars = 0
    chunks = 0
    try:
        with post_json(endpoint, payload, timeout) as response:
            if payload.get("stream", True):
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    now = time.perf_counter()
                    if first_chunk is None:
                        first_chunk = now
                    raw_data = line.removeprefix("data:").strip()
                    if not raw_data:
                        continue
                    chunks += 1
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        output_chars += len(delta.get("content") or "")
                        output_chars += len(delta.get("reasoning_content") or "")
            else:
                data = json.loads(response.read().decode("utf-8"))
                first_chunk = time.perf_counter()
                for choice in data.get("choices", []):
                    message = choice.get("message", {})
                    output_chars += len(message.get("content") or "")
                    output_chars += len(message.get("reasoning_content") or "")
                chunks = 1
        end = time.perf_counter()
        return {
            "ok": True,
            "ttft_seconds": None if first_chunk is None else first_chunk - start,
            "total_seconds": end - start,
            "output_chars": output_chars,
            "chunks": chunks,
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        end = time.perf_counter()
        return {
            "ok": False,
            "ttft_seconds": None,
            "total_seconds": end - start,
            "output_chars": 0,
            "chunks": 0,
            "error": f"{repr(exc)} {error_body}".strip(),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        end = time.perf_counter()
        return {
            "ok": False,
            "ttft_seconds": None,
            "total_seconds": end - start,
            "output_chars": 0,
            "chunks": 0,
            "error": repr(exc),
        }


def summarize(case: dict, raw_rows: list[dict]) -> dict:
    ok_rows = [row for row in raw_rows if row["ok"]]
    ttft = [row["ttft_seconds"] for row in ok_rows if row["ttft_seconds"] is not None]
    total = [row["total_seconds"] for row in ok_rows]
    chars = [row["output_chars"] for row in ok_rows]
    chunks = [row["chunks"] for row in ok_rows]
    return {
        **case,
        "requests": len(raw_rows),
        "errors": len(raw_rows) - len(ok_rows),
        "median_ttft_ms": None if not ttft else statistics.median(ttft) * 1000,
        "p90_ttft_ms": None if not ttft else percentile(ttft, 0.9) * 1000,
        "median_total_ms": None if not total else statistics.median(total) * 1000,
        "p90_total_ms": None if not total else percentile(total, 0.9) * 1000,
        "median_output_chars": None if not chars else statistics.median(chars),
        "median_chunks": None if not chunks else statistics.median(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniMind OpenAI-compatible API latency.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8998/v1/chat/completions")
    parser.add_argument("--model", default="minimind")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_chars", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests_per_case", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "api_latency_raw.jsonl"
    summary_path = args.output_dir / "api_latency_summary.json"
    csv_path = args.output_dir / "api_latency_summary.csv"

    summaries = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for prompt_chars in args.prompt_chars:
            prompt = make_prompt(prompt_chars)
            for concurrency in args.concurrency:
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "stream": args.stream,
                }
                case = {
                    "prompt_chars": prompt_chars,
                    "concurrency": concurrency,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stream": args.stream,
                }
                total_requests = concurrency * args.requests_per_case
                rows = []
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(run_one, args.endpoint, payload, args.timeout) for _ in range(total_requests)]
                    for future in as_completed(futures):
                        row = {**case, **future.result()}
                        rows.append(row)
                        raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        raw_handle.flush()
                summaries.append(summarize(case, rows))

    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(csv_path)


if __name__ == "__main__":
    main()
