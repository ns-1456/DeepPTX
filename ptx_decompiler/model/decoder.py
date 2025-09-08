"""AST Decoder with cross-attention to encoder and optional copy mechanism."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ptx_decompiler.model.positional import RotaryEmbedding, apply_rotary_pos_emb


def _rope_self_attn(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int,
    head_dim: int,
    causal_mask: bool,
    dropout: nn.Module,
) -> torch.Tensor:
    B, S, D = x.shape
    q = k = v = x
    q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, n_heads, head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    if causal_mask:
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = attn.softmax(dim=-1)
    attn = dropout(attn)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
    return out


def _cross_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    n_heads: int,
    head_dim: int,
    key_padding_mask: Optional[torch.Tensor],
    dropout: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = q.shape
    _, S, _ = kv.shape
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = kv.view(B, S, n_heads, head_dim).transpose(1, 2)
    v = kv.view(B, S, n_heads, head_dim).transpose(1, 2)
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    if key_padding_mask is not None:
        attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    attn_weights = attn.softmax(dim=-1)
    attn_weights = dropout(attn_weights)
    out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, D)
    return out, attn_weights


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block: self-attn (causal, RoPE) -> cross-attn -> FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.ln1(x)
        x = residual + _rope_self_attn(
            x_norm, cos, sin, self.n_heads, self.head_dim, causal_mask=True, dropout=self.dropout
        )
        residual = x
        x_norm = self.ln2(x)
        cross_out, attn_weights = _cross_attn(
            x_norm, memory, self.n_heads, self.head_dim,
            memory_key_padding_mask, self.dropout,
        )
        x = residual + cross_out
        x = x + self.ff(self.ln3(x))
        return x, attn_weights


class ASTDecoder(nn.Module):
    """Decoder for AST. Returns logits and cross-attention weights for copy mechanism."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.embed(tgt_ids))
        seq_len = x.size(1)
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        all_cross_attn = []
        for layer in self.layers:
            x, cross_attn = layer(x, memory, cos, sin, memory_key_padding_mask)
            all_cross_attn.append(cross_attn)
        x = self.ln_f(x)
        cross_attn_weights = all_cross_attn[-1]
        return x, cross_attn_weights
