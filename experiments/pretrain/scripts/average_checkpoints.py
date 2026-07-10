import argparse
import json
from pathlib import Path

import torch


def parse_item(text: str) -> tuple[Path, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("checkpoint item must be PATH=WEIGHT")
    path_text, weight_text = text.rsplit("=", 1)
    weight = float(weight_text)
    if weight < 0:
        raise argparse.ArgumentTypeError("weight must be non-negative")
    return Path(path_text), weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Average MiniMind state-dict checkpoints.")
    parser.add_argument("--checkpoint", action="append", type=parse_item, required=True, help="PATH=WEIGHT. Can be repeated.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    items = args.checkpoint
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0:
        raise SystemExit("sum of weights must be positive")

    averaged = None
    key_order = None
    manifest_items = []
    for path, raw_weight in items:
        if not path.exists():
            raise FileNotFoundError(path)
        weight = raw_weight / total_weight
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if averaged is None:
            key_order = list(state.keys())
            averaged = {key: state[key].float().mul(weight) for key in key_order}
        else:
            if list(state.keys()) != key_order:
                raise ValueError(f"checkpoint keys do not match: {path}")
            for key in key_order:
                averaged[key].add_(state[key].float(), alpha=weight)
        manifest_items.append({"path": str(path), "raw_weight": raw_weight, "normalized_weight": weight})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.half() for key, value in averaged.items()}, args.output)

    manifest = {
        "output": str(args.output),
        "checkpoints": manifest_items,
        "dtype": "float16",
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
