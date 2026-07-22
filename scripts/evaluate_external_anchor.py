"""Port an upstream MiniMind release checkpoint onto this project's model.

The external anchor is deliberately a different architecture from our candidate
-- that is the point of an external anchor. The only comparison it licenses is
loss and generation health *under the same tokenizer*. That last condition is
the one that decides whether the anchor means anything at all, so this script
refuses to emit a checkpoint unless the tokenizer the evaluation will use is
byte-identical to the tokenizer the upstream weights were trained with.

Two upstream facts drive the port:

* The released MiniMind2 weights predate upstream's QK normalization (their
  state dict has no ``q_norm``/``k_norm`` entries, added upstream in commit
  101d7df). The emitted checkpoint therefore carries ``qk_norm=False``, so the
  ordinary loader rebuilds it without those norms. Leaving them in would push
  trained weights through an untrained RMSNorm -- a different function from the
  one that was trained, and therefore a *fake* anchor rather than a weak one.
* Everything else matches: half-split NeoX RoPE at theta 1e6, pre-norm blocks,
  SwiGLU MLP, grouped-query attention with consecutive KV repeats, tied
  embeddings. ``--upstream-module`` proves this rather than assuming it, by
  running upstream's own module on the same input and comparing logits.

Usage::

    # convert (fails closed if the tokenizer does not match the weights)
    python scripts/evaluate_external_anchor.py \
        --checkpoint external-anchors/pretrain_768.pth \
        --tokenizer data/raw/minimind_official_v1/tokenizer \
        --output artifacts/anchors/upstream_104m.pt

    # prove the port reproduces upstream's own forward pass
    python scripts/evaluate_external_anchor.py ... --upstream-module /tmp/model_minimind.py

The emitted checkpoint is an ordinary internal checkpoint, so the anchor is
scored by the normal pipeline with no special-casing::

    python scripts/evaluate_candidate.py --checkpoint artifacts/anchors/upstream_104m.pt ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.model import ModelConfig, TransformerLM

# tokenizer.json shipped with gongjy/MiniMind2 and gongjy/MiniMind2-PyTorch on
# ModelScope, byte-identical to model/tokenizer.json at upstream commit 83e52f6.
# Upstream replaced this tokenizer wholesale in commit 101d7df ("minimind-3"):
# same 6400-token budget, but 1772 tokens swapped and 4619 tokens reassigned to
# different ids. Weights trained against one are meaningless under the other.
UPSTREAM_TOKENIZER_SHA256 = "d98595c6aef70d95f72748582fb9b4f53d76dd58c1ae1dd702ad7c84e1caf5e4"

# Upstream config values that are not recoverable from tensor shapes alone.
# Source: gongjy/MiniMind2 config.json (ModelScope, revision master).
UPSTREAM_NUM_ATTENTION_HEADS = 8
UPSTREAM_RMS_NORM_EPS = 1e-5
UPSTREAM_ROPE_THETA = 1_000_000.0
UPSTREAM_MAX_POSITION_EMBEDDINGS = 32768

_TOP_LEVEL_RENAMES = {
    "model.embed_tokens.weight": "token_embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

_PER_LAYER_RENAMES = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "self_attn.q_proj": "attn.q_proj",
    "self_attn.k_proj": "attn.k_proj",
    "self_attn.v_proj": "attn.v_proj",
    "self_attn.o_proj": "attn.o_proj",
    "mlp.gate_proj": "ffn.w1",
    "mlp.up_proj": "ffn.w2",
    "mlp.down_proj": "ffn.w3",
}

_LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)\.weight")


def remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename upstream parameters onto our module paths, refusing anything unknown.

    Raising on an unrecognised source key makes the mapping exhaustive from the
    upstream side; ``load_state_dict(strict=True)`` makes it exhaustive from
    ours. A silently dropped tensor would leave a randomly initialised layer in
    an otherwise trained model, which reads as a plausible-but-wrong loss.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in _TOP_LEVEL_RENAMES:
            remapped[_TOP_LEVEL_RENAMES[key]] = value
            continue
        match = _LAYER_KEY.fullmatch(key)
        target = _PER_LAYER_RENAMES.get(match.group(2)) if match else None
        if match is None or target is None:
            raise ValueError(f"unrecognised upstream checkpoint key: {key}")
        remapped[f"blocks.{match.group(1)}.{target}.weight"] = value
    return remapped


def infer_config(
    state_dict: dict[str, torch.Tensor],
    *,
    n_head: int = UPSTREAM_NUM_ATTENTION_HEADS,
    rms_norm_eps: float = UPSTREAM_RMS_NORM_EPS,
    rope_theta: float = UPSTREAM_ROPE_THETA,
    max_seq_len: int = UPSTREAM_MAX_POSITION_EMBEDDINGS,
) -> ModelConfig:
    """Derive our model configuration from the upstream tensor shapes."""
    vocab_size, n_embed = state_dict["model.embed_tokens.weight"].shape
    layer_indices = [
        int(match.group(1)) for key in state_dict if (match := _LAYER_KEY.fullmatch(key))
    ]
    if not layer_indices:
        raise ValueError("upstream checkpoint contains no transformer layers")

    query_shape = tuple(state_dict["model.layers.0.self_attn.q_proj.weight"].shape)
    if query_shape != (n_embed, n_embed):
        raise ValueError(
            f"q_proj is {query_shape}, expected {(n_embed, n_embed)}; our attention only "
            f"expresses head_dim = n_embed // n_head"
        )
    head_dim = n_embed // n_head
    kv_rows = int(state_dict["model.layers.0.self_attn.k_proj.weight"].shape[0])
    if kv_rows % head_dim:
        raise ValueError(f"k_proj rows {kv_rows} are not a multiple of head_dim {head_dim}")

    return ModelConfig(
        vocab_size=int(vocab_size),
        n_embed=int(n_embed),
        n_layer=max(layer_indices) + 1,
        n_head=n_head,
        n_kv_head=kv_rows // head_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,
        hidden_dim=int(state_dict["model.layers.0.mlp.gate_proj.weight"].shape[0]),
        tie_weights=True,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        # Recorded in the emitted checkpoint, so the ordinary loader rebuilds
        # the anchor without QK norm and no evaluation code needs special cases.
        qk_norm=False,
    )


def build_anchor_model(
    state_dict: dict[str, torch.Tensor],
    **config_overrides: Any,
) -> tuple[TransformerLM, ModelConfig]:
    """Load upstream weights into our model, exhaustively and in float32."""
    config = infer_config(state_dict, **config_overrides)
    model = TransformerLM(config)
    remapped = {name: value.float() for name, value in remap_state_dict(state_dict).items()}
    model.load_state_dict(remapped, strict=True)
    return model.eval(), config


def max_logit_difference(
    state_dict: dict[str, torch.Tensor],
    upstream_module_path: Path,
    *,
    seq_len: int = 128,
    batch_size: int = 2,
    seed: int = 20260721,
) -> float:
    """Run upstream's own module against our port and return the worst logit gap.

    This is the only evidence that the port is faithful. A plausible loss is
    not evidence: a subtly wrong RoPE convention or KV-repeat order still
    produces a finite, reasonable-looking number.
    """
    spec = importlib.util.spec_from_file_location("minimind_upstream", upstream_module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import upstream module from {upstream_module_path}")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    ours, config = build_anchor_model(state_dict)
    reference = upstream.MiniMindForCausalLM(
        upstream.MiniMindConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.n_embed,
            num_hidden_layers=config.n_layer,
            num_attention_heads=config.n_head,
            num_key_value_heads=config.n_kv_head,
            intermediate_size=config.hidden_dim,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            dropout=0.0,
        )
    )
    reference.load_state_dict({name: value.float() for name, value in state_dict.items()})
    reference.eval().float()

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long
    )
    with torch.inference_mode():
        reference_logits = reference(input_ids=input_ids).logits
        our_logits, _ = ours(input_ids)
    return float((reference_logits - our_logits).abs().max().item())


def sha256_tree(path: Path) -> dict[str, str]:
    """Hash every file under ``path`` by repository-relative name."""
    files = sorted(path.rglob("*") if path.is_dir() else [path])
    digests = {}
    for item in files:
        if item.is_file():
            digests[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
    return digests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="upstream .pth weights")
    parser.add_argument("--output", required=True, type=Path, help="internal checkpoint to write")
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="tokenizer directory the anchor will be evaluated with",
    )
    parser.add_argument(
        "--expect-tokenizer-sha256",
        default=UPSTREAM_TOKENIZER_SHA256,
        help="sha256 of tokenizer.json the upstream weights were trained with",
    )
    parser.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        help=(
            "emit the checkpoint even though the tokenizer differs. The resulting loss is "
            "NOT an anchor and must never be compared against our candidates; the only "
            "legitimate use is measuring how badly the mismatch degrades the model."
        ),
    )
    parser.add_argument(
        "--upstream-module",
        type=Path,
        help="path to upstream model_minimind.py, to prove the port reproduces its logits",
    )
    parser.add_argument("--max-logit-difference", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    tokenizer_json = args.tokenizer / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise SystemExit(f"no tokenizer.json under {args.tokenizer}")
    tokenizer_sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    tokenizer_matches = tokenizer_sha256 == args.expect_tokenizer_sha256
    if not tokenizer_matches and not args.allow_tokenizer_mismatch:
        raise SystemExit(
            "tokenizer mismatch: this anchor is void.\n"
            f"  evaluation tokenizer : {tokenizer_sha256}\n"
            f"  weights were trained with: {args.expect_tokenizer_sha256}\n"
            "Loss under a different tokenizer is not comparable, whatever it prints. "
            "Pass --allow-tokenizer-mismatch only to quantify the damage, never to anchor."
        )

    model, config = build_anchor_model(state_dict)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    verification: dict[str, Any] | None = None
    if args.upstream_module is not None:
        difference = max_logit_difference(state_dict, args.upstream_module)
        verification = {
            "upstream_module": str(args.upstream_module),
            "max_absolute_logit_difference": difference,
            "tolerance": args.max_logit_difference,
            "faithful": difference <= args.max_logit_difference,
        }
        if not verification["faithful"]:
            raise SystemExit(
                f"port is NOT faithful: max |logit difference| {difference:.3e} exceeds "
                f"{args.max_logit_difference:.1e}. A wrong anchor is worse than no anchor."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "external_anchor": {
                "source_checkpoint": args.checkpoint.name,
                "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "parameter_count": parameter_count,
                "qk_norm_removed": True,
                "tokenizer_sha256": tokenizer_sha256,
                "tokenizer_matches_weights": tokenizer_matches,
                "verification": verification,
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parameter_count": parameter_count,
                "config": config.to_dict(),
                "tokenizer_sha256": tokenizer_sha256,
                "tokenizer_matches_weights": tokenizer_matches,
                "tokenizer_files": sha256_tree(args.tokenizer),
                "verification": verification,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
