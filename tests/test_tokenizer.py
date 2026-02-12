"""Tests for PTX and AST tokenizers."""

import pytest
from ptx_decompiler.tokenizer import PTXTokenizer, ASTTokenizer
from ptx_decompiler.tokenizer.ast_tokenizer import tokenize_ast_sexp
from ptx_decompiler.tokenizer.ptx_tokenizer import tokenize_ptx_text


def test_ast_tokenize_sexp():
    tokens = tokenize_ast_sexp("(ADD (MUL A B) C)")
    assert "(" in tokens
    assert "ADD" in tokens
    assert ")" in tokens
    assert "A" in tokens


def test_ast_tokenizer_encode_decode():
    tok = ASTTokenizer()
    s = "(ADD A B)"
    ids = tok.encode(s, add_bos_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids, skip_special=True)
    assert "ADD" in decoded and "A" in decoded and "B" in decoded


def test_ptx_tokenizer_build_and_encode():
    tok = PTXTokenizer()
    tok.build_vocab(["ld.global.f32 %f0 [%r0]", "add.f32 %f1 %f0 %f2"])
    ids = tok.encode("ld.global.f32 %f0 [%r0]", add_bos_eos=True)
    assert len(ids) >= 3
    decoded = tok.decode(ids, skip_special=True)
    assert "ld" in decoded or "f32" in decoded


def test_ptx_tokenizer_special_ids():
    tok = PTXTokenizer()
    tok.build_vocab(["a b c"])
    assert tok.pad_id >= 0
    assert tok.bos_id >= 0
    assert tok.eos_id >= 0
