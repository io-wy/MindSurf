"""PyTorch datasets for language model training."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for pretraining/supervised fine-tuning from text files.

    Loads raw text, tokenizes, and creates fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length

        path = Path(data_path)
        if path.is_file():
            texts = [path.read_text(encoding="utf-8")]
        else:
            texts = [f.read_text(encoding="utf-8") for f in sorted(path.glob("*.txt"))]

        self.tokens = []
        for text in texts:
            self.tokens.extend(tokenizer.encode(text, add_special_tokens=False))

        # Build samples
        self.samples: list[list[int]] = []
        for i in range(0, len(self.tokens) - max_length, self.stride):
            self.samples.append(self.tokens[i : i + max_length + 1])

        if not self.samples:
            raise ValueError(f"No samples created from {data_path}. Text too short?")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = torch.tensor(sample[:-1], dtype=torch.long)
        labels = torch.tensor(sample[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class JsonlPackedDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream JSONL records and deterministically pack them into token blocks.

    The absolute packed-block cursor is the resume boundary. Training uses a
    single data-loader worker so the cursor has one unambiguous ordering.

    ``epochs`` repeats the file so a training budget can exceed one pass over
    the corpus. The cursor stays absolute across epoch boundaries, and the
    partial block at the end of an epoch carries into the next one, so exact
    resume is unaffected by where an interruption lands.

    ponytail: tokenising inline keeps the cursor definition trivial, at the cost
    of one saturated core per run while the rest of the host idles, roughly 6.5%
    of wall clock spent in data wait, and re-tokenising the whole corpus on every
    epoch. Measured on a 28-core host: training processes sat at 109% and 91%
    CPU with 390s and 342s of accumulated data wait. Upgrade path is to
    pre-tokenise once into a memory-mapped uint16 block array (vocab 6,400 fits,
    2.16B tokens costs 4.3 GB) and have this class slice it, which keeps
    ``num_workers=0`` and the absolute cursor while removing the bottleneck.
    Do this before the next long run.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        *,
        text_key: str = "text",
        skip_blocks: int = 0,
        max_blocks: int | None = None,
        shuffle_buffer: int = 0,
        seed: int = 20260511,
        epochs: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.skip_blocks = skip_blocks
        self.max_blocks = max_blocks
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epochs = epochs

        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.skip_blocks < 0:
            raise ValueError("skip_blocks must be non-negative")
        if self.shuffle_buffer < 0:
            raise ValueError("shuffle_buffer must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")

    def set_skip_blocks(self, skip_blocks: int) -> None:
        """Move the absolute resume cursor before constructing an iterator."""
        if skip_blocks < 0:
            raise ValueError("skip_blocks must be non-negative")
        self.skip_blocks = skip_blocks

    def _records(self, epoch: int = 0) -> Iterator[str]:
        with self.data_path.open("r", encoding="utf-8") as handle:
            if self.shuffle_buffer <= 1:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        text = row[self.text_key]
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"invalid JSONL record at {self.data_path}:{line_number}"
                        ) from exc
                    if not isinstance(text, str):
                        raise ValueError(
                            f"{self.text_key!r} must be a string at {self.data_path}:{line_number}"
                        )
                    yield text
                return

            rng = random.Random(self.seed + epoch)
            buffer: list[str] = []
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row[self.text_key]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid JSONL record at {self.data_path}:{line_number}"
                    ) from exc
                if not isinstance(text, str):
                    raise ValueError(
                        f"{self.text_key!r} must be a string at {self.data_path}:{line_number}"
                    )
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(text)
                    continue
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = text

            while buffer:
                yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "JsonlPackedDataset requires DataLoader(num_workers=0) "
                "to preserve an exact resume cursor"
            )

        eos_token_id = self.tokenizer.eos_token_id
        assert eos_token_id is not None
        block_size = self.max_length + 1
        token_buffer: list[int] = []
        absolute_block = 0
        emitted = 0

        for epoch in range(self.epochs):
            for text in self._records(epoch):
                token_buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
                token_buffer.append(eos_token_id)

                while len(token_buffer) >= block_size:
                    sample = token_buffer[:block_size]
                    del token_buffer[:block_size]

                    if absolute_block < self.skip_blocks:
                        absolute_block += 1
                        continue
                    if self.max_blocks is not None and emitted >= self.max_blocks:
                        return

                    absolute_block += 1
                    emitted += 1
                    values = torch.tensor(sample, dtype=torch.long)
                    yield {
                        "input_ids": values[:-1],
                        "labels": values[1:],
                    }


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for supervised fine-tuning with prompt-response pairs.

    Expects a JSONL file where each line is:
    {"prompt": "...", "response": "..."}
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        prompt_template: str = "### Instruction:\n{prompt}\n\n### Response:\n",
    ) -> None:
        super().__init__()
        import json

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

        path = Path(data_path)
        self.samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = self.prompt_template.format(prompt=sample["prompt"])
        response = sample["response"]

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_text = prompt + response + self.tokenizer.eos_token
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[: self.max_length]

        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        labels = input_ids.clone()

        # Mask prompt tokens in labels (only compute loss on response)
        labels[: len(prompt_tokens)] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader with padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = 0  # Will be overridden by tokenizer.pad_token_id in practice

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]

    return {"input_ids": input_ids, "labels": labels}


class PackedBlockDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream pre-tokenised blocks from a memory-mapped uint16 array.

    Same contract as :class:`JsonlPackedDataset` — absolute block cursor, one
    worker, deterministic order — but the tokenisation already happened, so the
    training process slices instead of parsing JSON and encoding text. That
    removes the saturated core and the data wait, and makes a second epoch cost
    nothing beyond re-reading the array.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        max_length: int,
        skip_blocks: int = 0,
        max_blocks: int | None = None,
        epochs: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.skip_blocks = skip_blocks
        self.max_blocks = max_blocks
        self.epochs = epochs

        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if skip_blocks < 0:
            raise ValueError("skip_blocks must be non-negative")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        self.block_size = max_length + 1
        item_size = np.dtype(np.uint16).itemsize
        size_bytes = self.data_path.stat().st_size
        if size_bytes % (self.block_size * item_size):
            raise ValueError(
                f"{self.data_path} is not a whole number of {self.block_size}-token blocks"
            )
        self.blocks_per_epoch = size_bytes // (self.block_size * item_size)

    def set_skip_blocks(self, skip_blocks: int) -> None:
        """Move the absolute resume cursor before constructing an iterator."""
        if skip_blocks < 0:
            raise ValueError("skip_blocks must be non-negative")
        self.skip_blocks = skip_blocks

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "PackedBlockDataset requires DataLoader(num_workers=0) "
                "to preserve an exact resume cursor"
            )

        flat = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        array = flat.reshape(-1, self.block_size)
        total = self.blocks_per_epoch * self.epochs
        for emitted, absolute in enumerate(range(self.skip_blocks, total)):
            if self.max_blocks is not None and emitted >= self.max_blocks:
                return
            # Wrapping by epoch keeps the cursor absolute across epoch
            # boundaries, exactly as the JSONL packer does.
            values = torch.from_numpy(array[absolute % self.blocks_per_epoch].astype(np.int64))
            yield {"input_ids": values[:-1], "labels": values[1:]}
