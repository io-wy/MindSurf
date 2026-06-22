import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return get_json(req, timeout)


def wait_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            get_json(f"{base_url}/healthz", timeout=3)
            return
        except Exception as exc:
            if time.time() > deadline:
                raise RuntimeError(f"healthz timeout: {exc}") from exc
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dynamic batching accepts mixed prompt lengths.")
    parser.add_argument("--base_url", default="http://127.0.0.1:8998")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--min_batch_size", type=int, default=2)
    args = parser.parse_args()

    wait_health(args.base_url, args.timeout)

    def post(repeats: int) -> dict:
        prompt = "mixed length batching check. " + ("context fragment. " * repeats)
        return post_json(
            f"{args.base_url}/v1/chat/completions",
            {
                "model": "minimind",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1.0,
                "max_tokens": 8,
                "stream": False,
            },
            args.timeout,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(post, n) for n in (1, 8, 16, 32)]):
            future.result()

    stats = get_json(f"{args.base_url}/healthz", timeout=5)["batch_stats"]
    if stats["max_batch_size"] < args.min_batch_size:
        raise AssertionError(stats)
    if len(set(stats["last_input_token_counts"])) < 2:
        raise AssertionError(stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
