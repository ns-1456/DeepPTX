"""Typed AST node hierarchy for CUDA expression trees. Serializes to S-expressions."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union

# --- Node types ---

@dataclass
class ASTNode(ABC):
    """Base class for all AST nodes."""

    def to_sexp(self) -> str:
        """Serialize to S-expression string."""
        raise NotImplementedError


@dataclass
class Var(ASTNode):
    """Variable reference: A, B, C, X, Y, etc."""
    name: str

    def to_sexp(self) -> str:
        return self.name


@dataclass
class Literal(ASTNode):
    """Numeric literal (float or int in AST)."""
    value: Union[float, int]

    def to_sexp(self) -> str:
        if isinstance(self.value, float):
            return str(self.value) if self.value != int(self.value) else f"{int(self.value)}.0"
        return str(self.value)


@dataclass
class BinOp(ASTNode):
    """Binary operation: ADD, SUB, MUL, DIV, MAX, MIN."""
    op: str  # ADD, SUB, MUL, DIV, MAX, MIN
    left: ASTNode
    right: ASTNode

    def to_sexp(self) -> str:
        return f"({self.op} {self.left.to_sexp()} {self.right.to_sexp()})"


@dataclass
class UnaryFunc(ASTNode):
    """Unary math function: SIN, COS, EXP, LOG, SQRT, RSQRT, NEG, ABS."""
    func: str
    arg: ASTNode

    def to_sexp(self) -> str:
        return f"({self.func} {self.arg.to_sexp()})"


@dataclass
class FMA(ASTNode):
    """Fused multiply-add: a * b + c."""
    a: ASTNode
    b: ASTNode
    c: ASTNode

    def to_sexp(self) -> str:
        return f"(FMA {self.a.to_sexp()} {self.b.to_sexp()} {self.c.to_sexp()})"


@dataclass
class Compare(ASTNode):
    """Comparison: GT, LT, GE, LE, EQ, NE."""
    op: str
    left: ASTNode
    right: ASTNode

    def to_sexp(self) -> str:
        return f"({self.op} {self.left.to_sexp()} {self.right.to_sexp()})"


@dataclass
class Ternary(ASTNode):
    """Ternary: condition ? if_true : if_false."""
    cond: Compare
    if_true: ASTNode
    if_false: ASTNode

    def to_sexp(self) -> str:
        return f"(TERNARY {self.cond.to_sexp()} {self.if_true.to_sexp()} {self.if_false.to_sexp()})"


# --- S-expression parser (for inference: string -> AST) ---

def parse_sexp(s: str) -> ASTNode:
    """Parse an S-expression string into an ASTNode tree."""
    s = s.strip()
    if not s:
        raise ValueError("Empty S-expression")

    # Variable or literal (single token)
    if s[0] != "(":
        token = s.split()[0] if s.split() else s
        if token in ("A", "B", "C", "X", "Y", "O", "T"):
            return Var(name=token)
        try:
            if "." in token:
                return Literal(value=float(token))
            return Literal(value=int(token))
        except ValueError:
            return Var(name=token)

    # (OP ...) or (FUNC arg) or (TERNARY c t f)
    if s[0] != "(" or s[-1] != ")":
        raise ValueError(f"Invalid S-expression: {s[:50]}...")
    inner = s[1:-1].strip()
    parts = _split_sexp(inner)

    if not parts:
        raise ValueError(f"Empty parens: {s}")

    op = parts[0].upper()
    args = parts[1:]

    if op in ("ADD", "SUB", "MUL", "DIV", "MAX", "MIN"):
        if len(args) != 2:
            raise ValueError(f"BinOp {op} needs 2 args, got {len(args)}")
        return BinOp(op=op, left=parse_sexp(args[0]), right=parse_sexp(args[1]))

    if op in ("SIN", "COS", "EXP", "LOG", "SQRT", "RSQRT", "NEG", "ABS"):
        if len(args) != 1:
            raise ValueError(f"UnaryFunc {op} needs 1 arg, got {len(args)}")
        return UnaryFunc(func=op, arg=parse_sexp(args[0]))

    if op == "FMA":
        if len(args) != 3:
            raise ValueError(f"FMA needs 3 args, got {len(args)}")
        return FMA(a=parse_sexp(args[0]), b=parse_sexp(args[1]), c=parse_sexp(args[2]))

    if op in ("GT", "LT", "GE", "LE", "EQ", "NE"):
        if len(args) != 2:
            raise ValueError(f"Compare {op} needs 2 args, got {len(args)}")
        return Compare(op=op, left=parse_sexp(args[0]), right=parse_sexp(args[1]))

    if op == "TERNARY":
        if len(args) != 3:
            raise ValueError(f"TERNARY needs 3 args, got {len(args)}")
        cond = parse_sexp(args[0])
        if not isinstance(cond, Compare):
            raise ValueError("TERNARY condition must be Compare")
        return Ternary(
            cond=cond,
            if_true=parse_sexp(args[1]),
            if_false=parse_sexp(args[2]),
        )

    raise ValueError(f"Unknown operator: {op}")


def _split_sexp(s: str) -> List[str]:
    """Split top-level S-expression into tokens (respecting nested parens)."""
    s = s.strip()
    result: List[str] = []
    i = 0
    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        if s[i] == "(":
            depth = 1
            start = i
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                i += 1
            result.append(s[start:i])
        else:
            start = i
            while i < len(s) and not s[i].isspace() and s[i] not in "()":
                i += 1
            if start < i:
                result.append(s[start:i])
    return result
