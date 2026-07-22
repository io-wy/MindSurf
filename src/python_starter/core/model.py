"""Lightweight decoder-only Transformer language model.

Inspired by minimind and modern LLM architectures (GPT/Llama style).
Features:
- Rotary Position Embeddings (RoPE)
- RMSNorm pre-normalization
- SwiGLU feed-forward network
- Configurable depth and width
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce in float32 and come back: under fp16 autocast both the mean of
        # squares and the addition of eps would otherwise happen at fp16, where
        # 1e-6 already sits in the subnormal range and flushes toward zero.
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm.type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.get_buffer("inv_freq")
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.outer(positions, inv_freq.to(x.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return (
            emb.cos().to(dtype=x.dtype)[None, None, :, :],
            emb.sin().to(dtype=x.dtype)[None, None, :, :],
        )


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention with QK normalization and RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embed % config.n_head != 0:
            raise ValueError("n_embed must be divisible by n_head")
        n_kv_head = config.n_kv_head or config.n_head
        if config.n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.n_head = config.n_head
        self.n_kv_head = n_kv_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        kv_dim = self.n_kv_head * self.head_dim
        self.k_proj = nn.Linear(config.n_embed, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embed, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        # Identity rather than a no-op flag: a disabled norm must contribute no
        # parameters, so loading such a checkpoint fails loudly on key mismatch
        # instead of quietly running trained weights through an untrained norm.
        self.q_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else nn.Identity()
        )

        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary(q, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if hasattr(functional, "scaled_dot_product_attention"):
            out = functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn = functional.softmax(scores, dim=-1)
            attn = functional.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embed)
        return cast(torch.Tensor, self.o_proj(out))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, n_embed: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * n_embed
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.dropout(self.w3(functional.silu(self.w1(x)) * self.w2(x))),
        )


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.n_embed, config.hidden_dim, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass(slots=True)
class ModelConfig:
    """Model hyperparameter configuration."""

    vocab_size: int = 6400
    n_embed: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int | None = None
    max_seq_len: int = 512
    dropout: float = 0.0
    hidden_dim: int | None = None
    tie_weights: bool = True
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    # Only external anchors set this false: upstream MiniMind's released
    # checkpoints predate its own QK normalization and carry no norm weights.
    qk_norm: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.n_embed <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size, n_embed, and n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        elif self.n_kv_head <= 0:
            raise ValueError("n_kv_head must be positive")
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.n_embed
        elif self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        # Zero divides by zero inside the norm; 1e-4 and above over-smooths and
        # costs final loss. The usable band is 1e-6 to 1e-5.
        if not 0 < self.rms_norm_eps < 1e-4:
            raise ValueError("rms_norm_eps must be in (0, 1e-4)")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe plain mapping.

        ``qk_norm`` is omitted while it holds its default so that checkpoints
        written before the field existed still compare equal to a freshly built
        configuration. Trainer resume tests the two mappings for exact equality,
        so emitting the key unconditionally would reject every prior checkpoint.
        """
        values = asdict(self)
        if values["qk_norm"]:
            del values["qk_norm"]
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        """Reconstruct a configuration stored in a checkpoint."""
        return cls(**values)


class TransformerLM(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds configured maximum {self.config.max_seq_len}"
            )
        x = self.token_embed(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len :]

            logits, _ = self(input_ids)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                continue

            logits = logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return input_ids
