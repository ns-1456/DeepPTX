"""AST Decoder with cross-attention to encoder and optional copy mechanism."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ptx_decompiler.model.positional import RotaryEmbedding, apply_rotary_pos_emb


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block: causal self-attn (RoPE) -> cross-attn -> FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # Self-attention projections
        self.self_q = nn.Linear(d_model, d_model)
        self.self_k = nn.Linear(d_model, d_model)
        self.self_v = nn.Linear(d_model, d_model)
        self.self_out = nn.Linear(d_model, d_model)

        # Cross-attention projections
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_out = nn.Linear(d_model, d_model)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def _causal_self_attention(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        q = self.self_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.self_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.self_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask: prevent attending to future tokens
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.self_out(out)

    def _cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        S = memory.size(1)

        q = self.cross_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # memory_key_padding_mask: (B, S), True = padding position to mask out
        if memory_key_padding_mask is not None:
            attn = attn.masked_fill(
                memory_key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = attn.softmax(dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)
        out = (attn_weights_dropped @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.cross_out(out), attn_weights  # return pre-dropout weights for copy

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Causal self-attention
        residual = x
        x = residual + self._causal_self_attention(self.ln1(x), cos, sin)

        # Cross-attention
        residual = x
        cross_out, attn_weights = self._cross_attention(
            self.ln2(x), memory, memory_key_padding_mask
        )
        x = residual + cross_out

        # FFN
        x = x + self.ff(self.ln3(x))
        return x, attn_weights


class ASTDecoder(nn.Module):
    """Decoder for AST. Returns hidden states and cross-attention weights for copy mechanism."""

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
        # Return last layer's cross-attention weights: (B, heads, T, S)
        cross_attn_weights = all_cross_attn[-1]
        return x, cross_attn_weights
