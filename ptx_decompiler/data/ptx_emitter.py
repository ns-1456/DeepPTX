"""
Pure-Python PTX emitter: AST -> synthetic PTX instructions (no nvcc needed).

Generates realistic PTX instruction sequences for CUDA kernels that compute
a single expression. Output goes through normalize_ptx() for consistency
with real nvcc output.

100k pairs in ~60 seconds instead of ~10 hours.
"""

import struct
from typing import Dict, Set

from ptx_decompiler.data.ast_nodes import (
    ASTNode, BinOp, Compare, FMA, Literal, Ternary, UnaryFunc, Var,
)

# Kernel parameter indices: k(float* A, float* B, float* C, float* X, float* Y, float* O, int N)
VAR_PARAM_IDX = {"A": 0, "B": 1, "C": 2, "X": 3, "Y": 4}
OUT_PARAM_IDX = 5
N_PARAM_IDX = 6

BINOP_PTX = {
    "ADD": "add.f32",
    "SUB": "sub.f32",
    "MUL": "mul.f32",
    "DIV": "div.rn.f32",
    "MAX": "max.f32",
    "MIN": "min.f32",
}

UNARY_PTX = {
    "SIN": "sin.approx.f32",
    "COS": "cos.approx.f32",
    "EXP": "ex2.approx.f32",
    "LOG": "lg2.approx.f32",
    "SQRT": "sqrt.rn.f32",
    "RSQRT": "rsqrt.approx.f32",
    "NEG": "neg.f32",
    "ABS": "abs.f32",
}

CMP_PTX = {"GT": "gt", "LT": "lt", "GE": "ge", "LE": "le", "EQ": "eq", "NE": "ne"}


def _float_to_hex(val: float) -> str:
    """Convert float to PTX hex immediate: 0fXXXXXXXX."""
    return "0f" + struct.pack(">f", val).hex().upper()


def _collect_vars(node: ASTNode) -> Set[str]:
    """Collect all variable names referenced in an expression."""
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Literal):
        return set()
    if isinstance(node, BinOp):
        return _collect_vars(node.left) | _collect_vars(node.right)
    if isinstance(node, UnaryFunc):
        return _collect_vars(node.arg)
    if isinstance(node, FMA):
        return _collect_vars(node.a) | _collect_vars(node.b) | _collect_vars(node.c)
    if isinstance(node, Compare):
        return _collect_vars(node.left) | _collect_vars(node.right)
    if isinstance(node, Ternary):
        return (
            _collect_vars(node.cond)
            | _collect_vars(node.if_true)
            | _collect_vars(node.if_false)
        )
    return set()


class PTXEmitter:
    """
    Emit PTX instruction sequences from AST nodes.

    Produces raw PTX lines (with semicolons and commas) that can be passed
    through normalize_ptx() for the same format as nvcc output.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._f_count = 0       # float registers: %f0, %f1, ...
        self._r_count = 5       # int registers: %r0-%r4 used by prologue
        self._rd_count = 0      # address registers: %rd0, %rd1, ...
        self._p_count = 1       # predicate registers: %p0 used by bounds check
        self._lines = []
        self._var_regs: Dict[str, str] = {}   # var name -> float register

    def _alloc_f(self) -> str:
        r = f"%f{self._f_count}"
        self._f_count += 1
        return r

    def _alloc_r(self) -> str:
        r = f"%r{self._r_count}"
        self._r_count += 1
        return r

    def _alloc_rd(self) -> str:
        r = f"%rd{self._rd_count}"
        self._rd_count += 1
        return r

    def _alloc_p(self) -> str:
        r = f"%p{self._p_count}"
        self._p_count += 1
        return r

    def _emit(self, line: str):
        self._lines.append(line)

    def _emit_prologue(self):
        """Thread index computation + bounds check. Uses %r0-%r4, %p0."""
        self._emit("mov.u32 %r0, %ctaid.x;")
        self._emit("mov.u32 %r1, %ntid.x;")
        self._emit("mov.u32 %r2, %tid.x;")
        self._emit("mad.lo.s32 %r3, %r0, %r1, %r2;")
        self._emit("ld.param.u32 %r4, [k_param_6];")
        self._emit("setp.ge.s32 %p0, %r3, %r4;")
        self._emit("@%p0 bra EXIT;")

    def _load_var(self, name: str) -> str:
        """Load array variable A[i], B[i], etc. Cached: each var loaded once."""
        if name in self._var_regs:
            return self._var_regs[name]
        param_idx = VAR_PARAM_IDX[name]
        rd_base = self._alloc_rd()
        rd_addr = self._alloc_rd()
        f_val = self._alloc_f()
        self._emit(f"ld.param.u64 {rd_base}, [k_param_{param_idx}];")
        self._emit(f"cvta.to.global.u64 {rd_base}, {rd_base};")
        self._emit(f"mul.wide.s32 {rd_addr}, %r3, 4;")
        self._emit(f"add.s64 {rd_addr}, {rd_base}, {rd_addr};")
        self._emit(f"ld.global.f32 {f_val}, [{rd_addr}];")
        self._var_regs[name] = f_val
        return f_val

    def _emit_store(self, result_reg: str):
        """Store result to O[i]."""
        rd_base = self._alloc_rd()
        rd_addr = self._alloc_rd()
        self._emit(f"ld.param.u64 {rd_base}, [k_param_{OUT_PARAM_IDX}];")
        self._emit(f"cvta.to.global.u64 {rd_base}, {rd_base};")
        self._emit(f"mul.wide.s32 {rd_addr}, %r3, 4;")
        self._emit(f"add.s64 {rd_addr}, {rd_base}, {rd_addr};")
        self._emit(f"st.global.f32 [{rd_addr}], {result_reg};")

    def _visit(self, node: ASTNode) -> str:
        """Recursively emit PTX for an expression. Returns register holding result."""
        if isinstance(node, Var):
            return self._load_var(node.name)

        if isinstance(node, Literal):
            r = self._alloc_f()
            hex_val = _float_to_hex(float(node.value))
            self._emit(f"mov.f32 {r}, {hex_val};")
            return r

        if isinstance(node, BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            r = self._alloc_f()
            self._emit(f"{BINOP_PTX[node.op]} {r}, {left}, {right};")
            return r

        if isinstance(node, UnaryFunc):
            arg = self._visit(node.arg)
            r = self._alloc_f()
            self._emit(f"{UNARY_PTX[node.func]} {r}, {arg};")
            return r

        if isinstance(node, FMA):
            a = self._visit(node.a)
            b = self._visit(node.b)
            c = self._visit(node.c)
            r = self._alloc_f()
            self._emit(f"fma.rn.f32 {r}, {a}, {b}, {c};")
            return r

        if isinstance(node, Ternary):
            cond_l = self._visit(node.cond.left)
            cond_r = self._visit(node.cond.right)
            pred = self._alloc_p()
            self._emit(
                f"setp.{CMP_PTX[node.cond.op]}.f32 {pred}, {cond_l}, {cond_r};"
            )
            true_r = self._visit(node.if_true)
            false_r = self._visit(node.if_false)
            r = self._alloc_f()
            self._emit(f"selp.f32 {r}, {true_r}, {false_r}, {pred};")
            return r

        if isinstance(node, Compare):
            # Standalone compare (shouldn't be at top level, but handle gracefully)
            left = self._visit(node.left)
            right = self._visit(node.right)
            r = self._alloc_f()
            self._emit(f"sub.f32 {r}, {left}, {right};")
            return r

        raise TypeError(f"Cannot emit PTX for node type: {type(node).__name__}")

    def emit(self, ast: ASTNode) -> str:
        """
        Emit raw PTX instruction lines for a kernel computing the given expression.
        Returns multi-line string (with semicolons, commas) ready for normalize_ptx().
        """
        self._reset()
        self._emit_prologue()
        result = self._visit(ast)
        self._emit_store(result)
        self._emit("EXIT:")
        self._emit("ret;")
        return "\n".join(self._lines)

    def emit_normalized(self, ast: ASTNode) -> str:
        """Emit and normalize in one call. Returns same format as normalize_ptx(nvcc_output)."""
        from ptx_decompiler.data.normalizer import normalize_ptx
        raw = self.emit(ast)
        return normalize_ptx(raw)
