"""PTX Encoder: embedding + RoPE + 6x Pre-Norm Transformer encoder layers."""

from typing import Optional

import torch
import torch.nn as nn

from ptx_decompiler.model.positional import RotaryEmbedding, apply_rotary_pos_emb


class PreNormEncoderBlock(nn.Module):
    """Pre-Norm Transformer encoder block: LN -> Self-Attn (RoPE) -> residual, LN -> FFN -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Q, K, V projections for self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x_norm = self.ln1(x)

        # Project Q, K, V
        q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q, K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # key_padding_mask: (B, S), True = padding position to mask out
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        x = residual + out
        x = x + self.ff(self.ln2(x))
        return x


class PTXEncoder(nn.Module):
    """Encoder for PTX sequences. Outputs memory for decoder."""

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
            PreNormEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # src_ids: (B, S_src), src_key_padding_mask: (B, S_src) True=padding
        x = self.dropout(self.embed(src_ids))
        seq_len = x.size(1)
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=src_key_padding_mask)
        return self.ln_f(x)
