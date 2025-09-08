"""Tests for grammar and AST nodes."""

import pytest
from ptx_decompiler.data.ast_nodes import (
    Var,
    Literal,
    BinOp,
    UnaryFunc,
    FMA,
    Compare,
    Ternary,
    parse_sexp,
)
from ptx_decompiler.data.grammar import (
    Tier1SimpleBinary,
    Tier2NestedArithmetic,
    Tier3UnaryMath,
    Tier4Ternary,
    get_tier_generator,
    sample_tier,
)


def test_var_to_sexp():
    assert Var(name="A").to_sexp() == "A"


def test_literal_to_sexp():
    assert Literal(value=1.0).to_sexp() in ("1.0", "1")
    assert Literal(value=0).to_sexp() == "0"


def test_binop_to_sexp():
    t = BinOp(op="ADD", left=Var("A"), right=Var("B"))
    assert t.to_sexp() == "(ADD A B)"


def test_parse_sexp_var():
    assert parse_sexp("A").to_sexp() == "A"


def test_parse_sexp_binop():
    t = parse_sexp("(ADD A B)")
    assert isinstance(t, BinOp)
    assert t.op == "ADD"
    assert t.left.to_sexp() == "A"
    assert t.right.to_sexp() == "B"


def test_parse_roundtrip():
    s = "(ADD (MUL A B) C)"
    t = parse_sexp(s)
    assert t.to_sexp() == s


def test_tier1_generate():
    gen = Tier1SimpleBinary()
    ast = gen.generate()
    assert ast.to_sexp().startswith("(")
    assert "ADD" in ast.to_sexp() or "MUL" in ast.to_sexp() or "SUB" in ast.to_sexp()


def test_tier2_generate():
    gen = Tier2NestedArithmetic()
    ast = gen.generate()
    sexp = ast.to_sexp()
    assert sexp.count("(") >= 2


def test_tier3_generate():
    gen = Tier3UnaryMath()
    ast = gen.generate()
    assert ast.to_sexp()


def test_tier4_generate():
    gen = Tier4Ternary()
    ast = gen.generate()
    assert "TERNARY" in ast.to_sexp()


def test_get_tier_generator():
    g = get_tier_generator(1)
    assert g.tier_id == 1
    with pytest.raises(ValueError):
        get_tier_generator(99)


def test_sample_tier():
    tier_id, gen = sample_tier()
    assert 1 <= tier_id <= 7
    assert gen.generate() is not None
