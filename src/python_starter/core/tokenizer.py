"""Tokenizer wrapper using the transformers library."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizer

from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def load_tokenizer(name_or_path: str = "gpt2") -> PreTrainedTokenizer:
    """Load a pretrained tokenizer.

    Args:
        name_or_path: HuggingFace model name or local path.

    Returns:
        Configured tokenizer instance.
    """
    logger.info("loading_tokenizer", name_or_path=name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    # Ensure special tokens exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def encode_text(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int | None = None,
    add_special_tokens: bool = True,
) -> list[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(
        text,
        max_length=max_length,
        truncation=max_length is not None,
        add_special_tokens=add_special_tokens,
    )


def decode_tokens(
    tokenizer: PreTrainedTokenizer,
    token_ids: list[int],
    skip_special_tokens: bool = True,
) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
