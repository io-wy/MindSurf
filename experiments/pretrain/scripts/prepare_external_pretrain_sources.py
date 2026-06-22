import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "flywheel_sources"
DEFAULT_QUALITY_CORE = DEFAULT_OUTPUT_DIR / "pretrain_quality_core_200000.jsonl"

DATASETS = {
    "tinystories_validation": {
        "repo": "roneneldan/TinyStories",
        "page": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "license": "cdla-sharing-1.0",
        "path": "default/validation/0000.parquet",
        "kind": "tinystories",
        "max_rows": 22000,
    },
    "gsm8k_train": {
        "repo": "openai/gsm8k",
        "page": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "mit",
        "path": "main/train/0000.parquet",
        "kind": "gsm8k",
        "max_rows": 8000,
    },
    "code_contests_valid": {
        "repo": "deepmind/code_contests",
        "page": "https://huggingface.co/datasets/deepmind/code_contests",
        "license": "cc-by-4.0",
        "path": "default/partial-valid/0000.parquet",
        "kind": "code_contests",
        "max_rows": 200,
    },
    "code_contests_test": {
        "repo": "deepmind/code_contests",
        "page": "https://huggingface.co/datasets/deepmind/code_contests",
        "license": "cc-by-4.0",
        "path": "default/partial-test/0000.parquet",
        "kind": "code_contests",
        "max_rows": 250,
    },
}

LANGUAGE_NAMES = {
    1: "python",
    2: "cpp",
    3: "python",
    4: "java",
    "PYTHON": "python",
    "CPP": "cpp",
    "PYTHON3": "python",
    "JAVA": "java",
}


def normalize(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def resolve_hf_parquet_url(repo: str, parquet_path: str) -> str:
    encoded_path = "/".join(quote(part) for part in parquet_path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{encoded_path}"


def download(url: str, path: Path, timeout: int = 60) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(path)


def load_rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def clean_tinystory(row: dict) -> str | None:
    text = normalize(row.get("text", ""))
    if len(text) < 120 or len(text) > 3500:
        return None
    if text.count("http://") or text.count("https://"):
        return None
    return text


def clean_gsm8k(row: dict) -> str | None:
    question = normalize(row.get("question", ""))
    answer = normalize(row.get("answer", ""))
    if len(question) < 20 or len(answer) < 20:
        return None
    return f"Math word problem:\nQuestion: {question}\nSolution: {answer}"


def choose_solution(solutions: dict) -> tuple[str, str] | None:
    if not isinstance(solutions, dict):
        return None
    languages = solutions.get("language") or []
    code_items = solutions.get("solution") or []
    if not code_items:
        return None
    preferred = {3: 0, "PYTHON3": 0, 1: 1, "PYTHON": 1, 2: 2, "CPP": 2, 4: 3, "JAVA": 3}
    ranked = []
    for index, code in enumerate(code_items):
        code = normalize(code)
        if len(code) < 80 or len(code) > 5000:
            continue
        language = languages[index] if index < len(languages) else ""
        ranked.append((preferred.get(language, 9), index, language, code))
    if not ranked:
        return None
    _, _, language, code = sorted(ranked)[0]
    return LANGUAGE_NAMES.get(language, "text"), code


def clean_code_contest(row: dict) -> str | None:
    description = normalize(row.get("description", ""))
    if len(description) < 200 or len(description) > 7000:
        return None
    picked = choose_solution(row.get("solutions", {}))
    if not picked:
        return None
    language, solution = picked
    return f"Programming problem:\n{description}\n\nReference solution:\n```{language}\n{solution}\n```"


def iter_clean_rows(dataset_key: str, rows: list[dict]):
    kind = DATASETS[dataset_key]["kind"]
    cleaner = {
        "tinystories": clean_tinystory,
        "gsm8k": clean_gsm8k,
        "code_contests": clean_code_contest,
    }[kind]
    max_rows = int(DATASETS[dataset_key]["max_rows"])
    count = 0
    for row in rows:
        text = cleaner(row)
        if not text:
            continue
        yield {"text": text}
        count += 1
        if count >= max_rows:
            break


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    return len(rows)


def rel(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return str(path)


def write_mix(path: Path, description: str, sources: list[tuple[str, Path, float]]) -> None:
    payload = {
        "description": description,
        "sources": [
            {"name": name, "path": rel(source), "weight": weight}
            for name, source, weight in sources
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare small external pretraining sources from HF parquet exports.")
    parser.add_argument("--cache_dir", type=Path, default=Path(r"D:\environment\cache\minimind-hf"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality_core", type=Path, default=DEFAULT_QUALITY_CORE)
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    quality_core = args.quality_core if args.quality_core.is_absolute() else ROOT / args.quality_core
    cache_dir = args.cache_dir

    written_sources = {}
    seen_hashes: set[str] = set()
    manifest = {
        "note": "External clean pretraining sources. Do not train on local_mcq_benchmark_v1 items.",
        "cache_dir": str(cache_dir),
        "datasets": {},
        "written": {},
        "mixes": [],
    }

    merged_code_rows = []
    for key, meta in DATASETS.items():
        url = resolve_hf_parquet_url(meta["repo"], meta["path"])
        parquet_path = cache_dir / meta["repo"].replace("/", "__") / meta["path"]
        if not args.skip_download:
            download(url, parquet_path)
        rows = load_rows(parquet_path)
        clean_rows = []
        for row in iter_clean_rows(key, rows):
            digest = text_hash(row["text"])
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            clean_rows.append(row)
        manifest["datasets"][key] = {
            "repo": meta["repo"],
            "page": meta["page"],
            "license": meta["license"],
            "parquet_url": url,
            "parquet_path": str(parquet_path),
            "raw_rows": len(rows),
            "clean_rows": len(clean_rows),
        }
        if meta["kind"] == "code_contests":
            merged_code_rows.extend(clean_rows)
            continue
        name = "external_tinystories" if meta["kind"] == "tinystories" else "external_gsm8k"
        path = output_dir / f"{name}_{len(clean_rows)}.jsonl"
        write_jsonl(path, clean_rows)
        written_sources[name] = path
        manifest["written"][name] = {"rows": len(clean_rows), "path": rel(path)}

    if merged_code_rows:
        path = output_dir / f"external_code_contests_{len(merged_code_rows)}.jsonl"
        write_jsonl(path, merged_code_rows)
        written_sources["external_code_contests"] = path
        manifest["written"]["external_code_contests"] = {"rows": len(merged_code_rows), "path": rel(path)}

    manifest["quality_core"] = {
        "path": rel(quality_core),
        "exists_on_this_machine": quality_core.exists(),
        "note": "The mix can still be valid on the server if this relative path exists there.",
    }
    english = written_sources["external_tinystories"]
    math = written_sources["external_gsm8k"]
    code = written_sources.get("external_code_contests")

    mix_specs = [
        (
            "mix_quality80_external_english10_math10.json",
            "Stage10 conservative external mix: quality80 + TinyStories English10 + GSM8K math10.",
            [("quality_core", quality_core, 0.80), ("external_tinystories", english, 0.10), ("external_gsm8k", math, 0.10)],
        ),
        (
            "mix_quality70_external_english15_math15.json",
            "Stage10 external mix: quality70 + TinyStories English15 + GSM8K math15.",
            [("quality_core", quality_core, 0.70), ("external_tinystories", english, 0.15), ("external_gsm8k", math, 0.15)],
        ),
    ]
    if code:
        mix_specs.append(
            (
                "mix_quality65_external_english15_math15_code5.json",
                "Stage10 external mix: quality65 + TinyStories English15 + GSM8K math15 + CodeContests code5.",
                [
                    ("quality_core", quality_core, 0.65),
                    ("external_tinystories", english, 0.15),
                    ("external_gsm8k", math, 0.15),
                    ("external_code_contests", code, 0.05),
                ],
            )
        )

    for filename, description, sources in mix_specs:
        path = output_dir / filename
        write_mix(path, description, sources)
        manifest["mixes"].append(rel(path))

    manifest_path = output_dir / "external_pretrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
