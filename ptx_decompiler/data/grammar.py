"""7-tier CUDA program grammar for curriculum data generation."""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Type

from ptx_decompiler.data.ast_nodes import (
    ASTNode,
    BinOp,
    Compare,
    FMA,
    Literal,
    Ternary,
    UnaryFunc,
    Var,
)

# Shared vocabulary
BINOPS = ["ADD", "SUB", "MUL", "DIV", "MAX", "MIN"]
UNARY_FUNCS = ["SIN", "COS", "EXP", "LOG", "SQRT", "RSQRT", "NEG", "ABS"]
COMPARE_OPS = ["GT", "LT", "GE", "LE", "EQ", "NE"]
VARS = ["A", "B", "C", "X", "Y"]
LITERAL_FLOATS = [0.0, 1.0, -1.0, 0.5, 2.0]


def _var() -> Var:
    return Var(name=random.choice(VARS))


def _literal() -> Literal:
    return Literal(value=random.choice(LITERAL_FLOATS))


class TierGenerator(ABC):
    """Base for tier-specific AST generators."""

    tier_id: int = 0
    complexity_score: float = 0.0

    @abstractmethod
    def generate(self) -> ASTNode:
        """Generate a random AST in this tier."""
        pass


class Tier1SimpleBinary(TierGenerator):
    """Tier 1: Simple binary ops only: (ADD A B), (MUL X Y)."""
    tier_id = 1
    complexity_score = 1.0

    def generate(self) -> ASTNode:
        op = random.choice(BINOPS)
        return BinOp(op=op, left=_var(), right=_var())


class Tier2NestedArithmetic(TierGenerator):
    """Tier 2: Nested arithmetic: (ADD (MUL A B) (SUB C X))."""
    tier_id = 2
    complexity_score = 2.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        if random.random() < 0.5:
            return _var()
        op = random.choice(BINOPS)
        return BinOp(
            op=op,
            left=self._expr(depth - 1),
            right=self._expr(depth - 1),
        )

    def generate(self) -> ASTNode:
        return self._expr(depth=2)


class Tier3UnaryMath(TierGenerator):
    """Tier 3: Unary math + FMA: (SIN A), (EXP (MUL A B)), (RSQRT X), (FMA A B C)."""
    tier_id = 3
    complexity_score = 3.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        r = random.random()
        if r < 0.4:
            return _var()
        if r < 0.7:
            op = random.choice(BINOPS)
            return BinOp(
                op=op,
                left=self._expr(depth - 1),
                right=self._expr(depth - 1),
            )
        if r < 0.9:
            func = random.choice(UNARY_FUNCS)
            return UnaryFunc(func=func, arg=self._expr(depth - 1))
        return FMA(
            a=self._expr(depth - 1),
            b=self._expr(depth - 1),
            c=self._expr(depth - 1),
        )

    def generate(self) -> ASTNode:
        return self._expr(depth=2)


class Tier4Ternary(TierGenerator):
    """Tier 4: Ternary / conditional: (TERNARY (GT A 0.0) (MUL A B) (NEG A))."""
    tier_id = 4
    complexity_score = 4.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        r = random.random()
        if r < 0.5:
            return _var()
        if r < 0.8:
            op = random.choice(BINOPS)
            return BinOp(
                op=op,
                left=self._expr(depth - 1),
                right=self._expr(depth - 1),
            )
        func = random.choice(UNARY_FUNCS)
        return UnaryFunc(func=func, arg=self._expr(depth - 1))

    def generate(self) -> ASTNode:
        cond_op = random.choice(COMPARE_OPS)
        cond = Compare(op=cond_op, left=_var(), right=_literal())
        return Ternary(
            cond=cond,
            if_true=self._expr(1),
            if_false=self._expr(1),
        )


class Tier5TypeDiversity(TierGenerator):
    """Tier 5: Same expressions but we tag type (float/int/half) for renderer; AST is same, type is metadata."""
    tier_id = 5
    complexity_score = 5.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        r = random.random()
        if r < 0.4:
            return _var()
        if r < 0.75:
            op = random.choice(BINOPS)
            return BinOp(
                op=op,
                left=self._expr(depth - 1),
                right=self._expr(depth - 1),
            )
        func = random.choice(UNARY_FUNCS)
        return UnaryFunc(func=func, arg=self._expr(depth - 1))

    def generate(self) -> ASTNode:
        return self._expr(depth=2)


class Tier6MultiStatement(TierGenerator):
    """Tier 6: Single complex expression (multi-statement handled in renderer as one output + temp)."""
    tier_id = 6
    complexity_score = 6.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        r = random.random()
        if r < 0.35:
            return _var()
        if r < 0.65:
            op = random.choice(BINOPS)
            return BinOp(
                op=op,
                left=self._expr(depth - 1),
                right=self._expr(depth - 1),
            )
        if r < 0.85:
            func = random.choice(UNARY_FUNCS)
            return UnaryFunc(func=func, arg=self._expr(depth - 1))
        cond_op = random.choice(COMPARE_OPS)
        cond = Compare(op=cond_op, left=_var(), right=_literal())
        return Ternary(
            cond=cond,
            if_true=self._expr(1),
            if_false=self._expr(1),
        )

    def generate(self) -> ASTNode:
        return self._expr(depth=3)


class Tier7SharedMemory(TierGenerator):
    """Tier 7: Single expression (shared-memory reduction is a future extension; for now same as Tier 6)."""
    tier_id = 7
    complexity_score = 7.0

    def _expr(self, depth: int) -> ASTNode:
        if depth <= 0:
            return _var()
        r = random.random()
        if r < 0.4:
            return _var()
        op = random.choice(BINOPS)
        return BinOp(
            op=op,
            left=self._expr(depth - 1),
            right=self._expr(depth - 1),
        )

    def generate(self) -> ASTNode:
        return self._expr(depth=2)


# Registry and weights (biased toward Tiers 1-4 for curriculum)
TIER_CLASSES: List[Type[TierGenerator]] = [
    Tier1SimpleBinary,
    Tier2NestedArithmetic,
    Tier3UnaryMath,
    Tier4Ternary,
    Tier5TypeDiversity,
    Tier6MultiStatement,
    Tier7SharedMemory,
]

# Weights for sampling: more mass on 1-4
TIER_WEIGHTS: List[float] = [0.25, 0.25, 0.20, 0.15, 0.07, 0.05, 0.03]


def get_tier_generator(tier_id: int) -> TierGenerator:
    """Return a generator instance for the given tier (1-7)."""
    for cls in TIER_CLASSES:
        if cls.tier_id == tier_id:
            return cls()
    raise ValueError(f"Unknown tier_id: {tier_id}")


def sample_tier() -> Tuple[int, TierGenerator]:
    """Sample a tier by TIER_WEIGHTS and return (tier_id, generator)."""
    cls = random.choices(TIER_CLASSES, weights=TIER_WEIGHTS, k=1)[0]
    return cls.tier_id, cls()
