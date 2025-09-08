"""Full PTX -> AST model: Encoder + Decoder + Copy Generator."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ptx_decompiler.model.encoder import PTXEncoder
from ptx_decompiler.model.decoder import ASTDecoder
from ptx_decompiler.model.copy_mechanism import CopyGenerator


class PTXDecompilerModel(nn.Module):
    """
    Encoder-decoder with pointer-generator.
    Encoder: PTX tokens -> memory.
    Decoder: AST tokens (shifted right) -> decoder_out + cross_attn_weights.
    CopyGenerator: logits = p_gen * P_vocab + (1-p_gen) * P_copy.
    ptx_to_ast_map: optional (ptx_vocab_size,) LongTensor; map[ptx_id] = ast_id or -1.
    """

    def __init__(
        self,
        ptx_vocab_size: int,
        ast_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_copy: bool = True,
        ptx_to_ast_map: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.use_copy = use_copy
        if ptx_to_ast_map is not None:
            self.register_buffer("ptx_to_ast_map", ptx_to_ast_map.long())
        else:
            self.register_buffer("ptx_to_ast_map", None)
        self.encoder = PTXEncoder(
            vocab_size=ptx_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=encoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.decoder = ASTDecoder(
            vocab_size=ast_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=decoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.copy_generator = CopyGenerator(d_model, ast_vocab_size, dropout) if use_copy else None
        if not use_copy:
            self.output_proj = nn.Linear(d_model, ast_vocab_size)

    def forward(
        self,
        ptx_ids: torch.Tensor,
        ast_input_ids: torch.Tensor,
        ptx_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ptx_ids: (B, S_src)
        ast_input_ids: (B, T) decoder input (target shifted right)
        ptx_padding_mask: (B, S_src) True where padding (to mask out)
        Returns: logits (B, T, ast_vocab_size), p_gen (B, T, 1) or None
        """
        if ptx_padding_mask is None:
            ptx_padding_mask = (ptx_ids == 0)
        memory = self.encoder(ptx_ids, src_key_padding_mask=ptx_padding_mask)
        decoder_out, cross_attn = self.decoder(
            ast_input_ids,
            memory,
            memory_key_padding_mask=ptx_padding_mask,
        )
        if self.use_copy and self.copy_generator is not None:
            cross_attn_flat = cross_attn.mean(dim=1)
            encoder_ast_ids = (
                self.ptx_to_ast_map[ptx_ids] if self.ptx_to_ast_map is not None
                else torch.full_like(ptx_ids, -1, device=ptx_ids.device)
            )
            logits, p_gen = self.copy_generator(
                decoder_out,
                cross_attn_flat,
                encoder_ast_ids,
                encoder_padding_mask=ptx_padding_mask,
            )
            return logits, p_gen
        logits = self.output_proj(decoder_out)
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
