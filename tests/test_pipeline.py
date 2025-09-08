"""Tests for renderer, normalizer, and pipeline components."""

import pytest
from ptx_decompiler.data import (
    parse_sexp,
    ast_to_cuda,
    normalize_ptx,
    CUDARenderer,
)
from ptx_decompiler.data.ast_nodes import BinOp, Var


def test_ast_to_cuda():
    cuda = ast_to_cuda("(ADD A B)")
    assert "__global__" in cuda
    assert "O[i]" in cuda
    assert "A[i]" in cuda
    assert "B[i]" in cuda


def test_renderer_binop():
    tree = BinOp(op="MUL", left=Var("X"), right=Var("Y"))
    r = CUDARenderer()
    src = r.kernel_source(tree)
    assert "X[i]" in src
    assert "Y[i]" in src
    assert "*" in src


def test_normalize_ptx_strips_directives():
    raw = """.version 7.0
.target sm_75
.visible .entry k(
ld.global.f32 %f1, [%rd1];
add.f32 %f2, %f1, %f0;
)"""
    out = normalize_ptx(raw)
    assert ".version" not in out
    assert "ld" in out or "add" in out


def test_parse_sexp_roundtrip():
    s = "(ADD (MUL A B) C)"
    t = parse_sexp(s)
    assert t.to_sexp() == s
