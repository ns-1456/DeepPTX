"""Tests for model forward pass."""

import torch
from ptx_decompiler.model import PTXDecompilerModel


def test_decompiler_forward():
    model = PTXDecompilerModel(
        ptx_vocab_size=100,
        ast_vocab_size=50,
        d_model=64,
        n_heads=4,
        d_ff=256,
        encoder_layers=2,
        decoder_layers=2,
        use_copy=False,
    )
    B, S, T = 2, 20, 15
    ptx_ids = torch.randint(0, 100, (B, S))
    ast_input = torch.randint(0, 50, (B, T))
    logits, p_gen = model(ptx_ids, ast_input, None)
    assert logits.shape == (B, T, 50)
    assert p_gen is None


def test_decompiler_forward_with_copy():
    ptx_vocab_size, ast_vocab_size = 100, 50
    ptx_to_ast = torch.full((ptx_vocab_size,), -1, dtype=torch.long)
    ptx_to_ast[10] = 5
    ptx_to_ast[11] = 6
    model = PTXDecompilerModel(
        ptx_vocab_size=ptx_vocab_size,
        ast_vocab_size=ast_vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=256,
        encoder_layers=2,
        decoder_layers=2,
        use_copy=True,
        ptx_to_ast_map=ptx_to_ast,
    )
    B, S, T = 2, 20, 15
    ptx_ids = torch.randint(0, 100, (B, S))
    ast_input = torch.randint(0, 50, (B, T))
    logits, p_gen = model(ptx_ids, ast_input, None)
    assert logits.shape == (B, T, ast_vocab_size)
    assert p_gen.shape == (B, T, 1)


def test_count_parameters():
    model = PTXDecompilerModel(
        ptx_vocab_size=100,
        ast_vocab_size=50,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
        use_copy=False,
    )
    n = model.count_parameters()
    assert n > 0
    assert n < 10_000_000
