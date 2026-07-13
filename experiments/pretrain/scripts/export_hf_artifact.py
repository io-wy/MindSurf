from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindModel


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    *,
    weight_path: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path,
    config: MiniMindConfig,
    source_revision: str | None,
) -> dict[str, Any]:
    source = Path(weight_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"weight file not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    state = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    source_dtype = next(
        (tensor.dtype for tensor in state.values() if torch.is_floating_point(tensor)),
        torch.float32,
    )
    model = MiniMindForCausalLM(config).to(dtype=source_dtype).eval()
    model.load_state_dict(state, strict=True)

    MiniMindConfig.register_for_auto_class()
    MiniMindModel.register_for_auto_class("AutoModel")
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model.config.auto_map = {
        "AutoConfig": "model_minimind.MiniMindConfig",
        "AutoModel": "model_minimind.MiniMindModel",
        "AutoModelForCausalLM": "model_minimind.MiniMindForCausalLM",
    }
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(destination)

    artifact_files = []
    for path in sorted(destination.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "source_revision": source_revision,
        "source_weight": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "files": artifact_files,
    }
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MiniMind .pth checkpoint as a self-contained Hugging Face artifact")
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--source_revision", default=None)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    args = parser.parse_args()
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
    )
    manifest = export_checkpoint(
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        config=config,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
