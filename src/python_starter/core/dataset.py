"""PyTorch datasets for language model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """Dataset for pretraining/supervised fine-tuning from text files.

    Loads raw text, tokenizes, and creates fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
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
            texts = [
                f.read_text(encoding="utf-8")
                for f in sorted(path.glob("*.txt"))
            ]

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


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with prompt-response pairs.

    Expects a JSONL file where each line is:
    {"prompt": "...", "response": "..."}
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
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
