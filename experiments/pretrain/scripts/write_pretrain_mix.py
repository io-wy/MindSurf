import argparse
import json
from pathlib import Path


def parse_source(text: str) -> dict:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("source must be NAME:PATH:WEIGHT")
    name, path, weight = parts
    weight_value = float(weight)
    if weight_value <= 0:
        raise argparse.ArgumentTypeError("source weight must be positive")
    return {"name": name, "path": path, "weight": weight_value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a weighted pretraining data mix JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description", default="Weighted pretraining mix.")
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    args = parser.parse_args()

    payload = {"description": args.description, "sources": args.source}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
