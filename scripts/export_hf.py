"""Export released weights as a stock Hugging Face model directory.

TransformerLM is Qwen3's architecture under different names: attention runs
q_proj, then a per-head RMSNorm, then RoPE, which is the arrangement Qwen3 uses
and Llama does not. So the export is a rename, not a reimplementation, and the
result loads with AutoModelForCausalLM without trust_remote_code.

A rename is exactly the kind of change that fails silently -- swap two FFN
matrices and the model still runs, just worse. So the export refuses to write
anything until the converted model reproduces the source model's logits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.model import ModelConfig, TransformerLM  # noqa: E402

PROBE_ROWS = 2
PROBE_LENGTH = 12


def probe_ids(vocab_size: int, seq_len: int) -> torch.Tensor:
    """Deterministic probe ids spread across the whole vocabulary.

    Fixed rather than random so a failure is reproducible, and spread so that a
    mismatch confined to one embedding range still shows up. Derived from the
    vocabulary rather than hard-coded, so the check works on any config.
    """
    length = min(PROBE_LENGTH, seq_len)
    step = max(1, vocab_size // length)
    ascending = [(index * step) % vocab_size for index in range(length)]
    descending = [(vocab_size - 1 - index * step) % vocab_size for index in range(length)]
    return torch.tensor([ascending, descending][:PROBE_ROWS], dtype=torch.long)


def build_state_dict(state: dict[str, torch.Tensor], n_layer: int) -> dict[str, torch.Tensor]:
    """Rename our parameters onto Qwen3's."""
    out: dict[str, torch.Tensor] = {"model.embed_tokens.weight": state["token_embed.weight"]}
    for index in range(n_layer):
        src = f"blocks.{index}."
        dst = f"model.layers.{index}."
        out[dst + "input_layernorm.weight"] = state[src + "attn_norm.weight"]
        out[dst + "post_attention_layernorm.weight"] = state[src + "ffn_norm.weight"]
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[dst + f"self_attn.{projection}.weight"] = state[src + f"attn.{projection}.weight"]
        out[dst + "self_attn.q_norm.weight"] = state[src + "attn.q_norm.weight"]
        out[dst + "self_attn.k_norm.weight"] = state[src + "attn.k_norm.weight"]
        # w1 is the branch that goes through silu, so it is the gate; w2 is the
        # linear branch. Swapping these two leaves a model that still runs.
        out[dst + "mlp.gate_proj.weight"] = state[src + "ffn.w1.weight"]
        out[dst + "mlp.up_proj.weight"] = state[src + "ffn.w2.weight"]
        out[dst + "mlp.down_proj.weight"] = state[src + "ffn.w3.weight"]
    out["model.norm.weight"] = state["norm.weight"]
    out["lm_head.weight"] = state["lm_head.weight"]
    return out


def build_config(config: ModelConfig) -> Any:
    from transformers import Qwen3Config

    if not config.qk_norm:
        raise ValueError(
            "this model has no QK normalisation, which Qwen3 requires; "
            "external anchors cannot be exported this way"
        )
    return Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.n_embed,
        intermediate_size=config.hidden_dim,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_kv_head,
        head_dim=config.n_embed // config.n_head,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=config.rms_norm_eps,
        # transformers 5 carries the RoPE base inside rope_parameters. The
        # top-level rope_theta a transformers 4 reader looks for is written
        # back into the saved config by _write_legacy_rope_theta.
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        tie_word_embeddings=config.tie_weights,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        use_sliding_window=False,
    )


def convert(config: ModelConfig, state: dict[str, torch.Tensor]) -> Any:
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM(build_config(config))  # type: ignore[no-untyped-call]
    model.load_state_dict(build_state_dict(state, config.n_layer), strict=True)
    return model.eval().float()  # type: ignore[no-untyped-call]


def max_logit_difference(source: TransformerLM, exported: Any) -> float:
    """Largest absolute logit gap between the two models on the probes."""
    input_ids = probe_ids(source.config.vocab_size, source.config.max_seq_len)
    with torch.inference_mode():
        ours, _ = source(input_ids, None)
        theirs = exported(input_ids).logits
    return float((ours.float() - theirs.float()).abs().max())


def _load_exported(directory: Path) -> Any:
    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    return loaded.eval()  # type: ignore[no-untyped-call]


def _write_legacy_rope_theta(directory: Path, rope_theta: float) -> None:
    """Also record rope_theta at the top level of the config.

    transformers 5 moved it under rope_parameters. A reader on transformers 4
    finds no rope_theta, silently falls back to 10000, and gets a model whose
    positions are wrong everywhere without a single error being raised.
    """
    path = directory / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.setdefault("rope_theta", rope_theta)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _adopt_tokenizer_special_ids(directory: Path, tokenizer_dir: Path) -> None:
    """Copy the tokenizer's special ids onto the model and generation configs.

    Without eos_token_id the generation loop has no stop condition and runs to
    max_new_tokens every time.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for name in ("config.json", "generation_config.json"):
        path = directory / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({key: value for key, value in ids.items() if value is not None})
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="released .pt export")
    parser.add_argument("--output", required=True, type=Path, help="directory to write")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer directory to copy alongside")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="largest tolerated logit difference; the default demands equality",
    )
    args = parser.parse_args()

    blob = torch.load(args.weights, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(blob["model_config"])
    state = blob["model_state_dict"]

    source = TransformerLM(config)
    source.load_state_dict(state, strict=True)
    source.eval().float()

    exported = convert(config, state)
    difference = max_logit_difference(source, exported)
    if difference > args.tolerance:
        raise SystemExit(
            f"export rejected: logits differ by {difference:.6e}, "
            f"above the tolerated {args.tolerance:.6e}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    exported.save_pretrained(args.output)
    if args.tokenizer is not None:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            source_file = args.tokenizer / name
            if source_file.is_file():
                shutil.copy2(source_file, args.output / name)
        _adopt_tokenizer_special_ids(args.output, args.tokenizer)
    _write_legacy_rope_theta(args.output, config.rope_theta)

    # What a reader loads is the directory, not the object in this process, so
    # the equivalence has to be re-checked after a round trip through disk: a
    # config field that fails to serialise would otherwise pass unnoticed.
    reloaded_difference = max_logit_difference(source, _load_exported(args.output))
    if reloaded_difference > args.tolerance:
        raise SystemExit(
            f"export rejected after reload: logits differ by {reloaded_difference:.6e}, "
            f"above the tolerated {args.tolerance:.6e}"
        )

    # The release block carries the licence, which must not be separable from
    # the weights it restricts.
    release = blob.get("release")
    if isinstance(release, dict):
        (args.output / "mindsurf_release.json").write_text(
            json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "architecture": "Qwen3ForCausalLM",
                "max_logit_difference": difference,
                "max_logit_difference_after_reload": reloaded_difference,
                "parameters": sum(p.numel() for p in exported.parameters()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
