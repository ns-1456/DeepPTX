"""Pointer-Generator: p_gen * P_vocab + (1 - p_gen) * P_copy."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CopyGenerator(nn.Module):
    """
    P(token) = p_gen * P_vocab + (1 - p_gen) * P_copy.
    P_copy is the attention distribution over encoder (PTX) tokens.
    Copy probs are scattered into vocab indices using encoder_token_ids.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj_vocab = nn.Linear(d_model, vocab_size)
        self.proj_gen = nn.Linear(d_model, 1)

    def forward(
        self,
        decoder_out: torch.Tensor,
        cross_attn_weights: torch.Tensor,
        encoder_token_ids: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_out: (B, T, D)
        cross_attn_weights: (B, T, S) from decoder cross-attention
        encoder_token_ids: (B, S) - source token ids
        Returns: logits (B, T, vocab_size), p_gen (B, T, 1)
        """
        B, T, _ = decoder_out.shape
        logits_vocab = self.proj_vocab(decoder_out)
        p_gen = torch.sigmoid(self.proj_gen(decoder_out))

        # P_copy: scatter copy prob into AST vocab indices (encoder_token_ids = AST vocab id per src pos, or -1)
        if encoder_padding_mask is not None:
            cross_attn_weights = cross_attn_weights.masked_fill(
                encoder_padding_mask.unsqueeze(1), 0.0
            )
        row_sum = cross_attn_weights.sum(dim=-1, keepdim=True) + 1e-9
        cross_attn_weights = cross_attn_weights / row_sum
        one_minus_p = (1 - p_gen).squeeze(-1)
        contribution = one_minus_p.unsqueeze(-1) * cross_attn_weights
        enc_ids = encoder_token_ids.unsqueeze(1).expand(B, T, -1)
        valid = (enc_ids >= 0) & (enc_ids < self.vocab_size)
        enc_ids_clamped = enc_ids.clamp(0, self.vocab_size - 1).long()
        contribution = contribution * valid.float()
        logits_vocab.scatter_add_(2, enc_ids_clamped, contribution)
        return logits_vocab, p_gen
