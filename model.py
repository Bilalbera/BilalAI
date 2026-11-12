"""BilalAI 1.0 - Decoder-only Transformer dil modeli.

Mimari, teknik sartname bolum 2 ile birebir uyumludur:
  vocab 32768, d_model 768, 13 katman, 12 head (64 dim), FFN 3072,
  max_seq_len 1024, RMSNorm (pre-norm), GELU, causal attention,
  learned position embedding, weight tying.

Benzersiz egitilebilir parametre sayisi: 118.075.392.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BilalAIConfig


class RMSNorm(nn.Module):
    """Bias'siz, tek ogrenilebilir olcek vektorlu RMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed


class CausalSelfAttention(nn.Module):
    """Cok head'li causal self-attention (tek QKV projeksiyonu)."""

    def __init__(self, config: BilalAIConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout

        # 768 -> 2304 (Q, K, V tek matriste)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.linear_bias)
        # 768 -> 768 output projeksiyonu
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=2)

        # [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        # causal_mask: [T, T], True = izin ver (j <= i)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if padding_mask is not None:
            # padding_mask: [B, T], True = gecerli token. Pad olan key'leri maskele.
            key_mask = padding_mask[:, None, None, :]  # [B, 1, 1, T]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class MLP(nn.Module):
    """768 -> 3072 (GELU) -> 768, her iki katmanda bias."""

    def __init__(self, config: BilalAIConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim, bias=config.linear_bias)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_model, bias=config.linear_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-normalization decoder blogu: x = x + attn(RMSNorm(x)); x = x + mlp(RMSNorm(x))."""

    def __init__(self, config: BilalAIConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask, padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BilalAIModel(nn.Module):
    """BilalAI 1.0 decoder-only Transformer."""

    def __init__(self, config: BilalAIConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # lm_head bias'siz; agirligi token_embedding ile paylasilir (weight tying).
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.embedding_tying:
            self.lm_head.weight = self.token_embedding.weight

        # Causal mask: alt ucgen (j <= i True). Egitilebilir degil, buffer.
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids: [B, T] (long). Cikti: logits [B, T, vocab_size]."""
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Girdi uzunlugu {T}, max_seq_len {self.config.max_seq_len} sinirini asiyor."
            )
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        causal_mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def num_unique_trainable_parameters(self) -> int:
        """requires_grad=True olan benzersiz parametreleri sayar (weight tying tek sayilir)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def build_model(config: BilalAIConfig | None = None) -> BilalAIModel:
    return BilalAIModel(config or BilalAIConfig())


if __name__ == "__main__":
    cfg = BilalAIConfig()
    model = build_model(cfg)
    n = model.num_unique_trainable_parameters()
    print(f"{cfg.model_name}: {n:,} egitilebilir parametre")