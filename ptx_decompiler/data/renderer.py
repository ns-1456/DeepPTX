"""AST to valid CUDA C++ source. Produces a __global__ kernel with correct indexing."""

from typing import Optional

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

# Default dtype for kernel (Tier 5 can override via metadata; we use float for simplicity)
DEFAULT_DTYPE = "float"

# C++ operator and function mapping
BINOP_CPP = {
    "ADD": "+",
    "SUB": "-",
    "MUL": "*",
    "DIV": "/",
    "MAX": "fmaxf",
    "MIN": "fminf",
}

UNARY_CPP = {
    "SIN": "sinf",
    "COS": "cosf",
    "EXP": "expf",
    "LOG": "logf",
    "SQRT": "sqrtf",
    "RSQRT": "rsqrtf",
    "NEG": "-",
    "ABS": "fabsf",
}

COMPARE_CPP = {
    "GT": ">",
    "LT": "<",
    "GE": ">=",
    "LE": "<=",
    "EQ": "==",
    "NE": "!=",
}


class CUDARenderer:
    """Renders ASTNode to CUDA C++ expression and full kernel source."""

    def __init__(self, dtype: str = DEFAULT_DTYPE):
        self.dtype = dtype

    def expr(self, node: ASTNode) -> str:
        """Convert a single expression node to C++ expression with [i] indexing."""
        if isinstance(node, Var):
            return f"{node.name}[i]"
        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, float):
                return f"{v}f" if v != int(v) else f"{int(v)}.0f"
            return str(v)
        if isinstance(node, BinOp):
            left = self.expr(node.left)
            right = self.expr(node.right)
            if node.op in ("MAX", "MIN"):
                return f"{BINOP_CPP[node.op]}({left}, {right})"
            return f"({left} {BINOP_CPP[node.op]} {right})"
        if isinstance(node, UnaryFunc):
            arg = self.expr(node.arg)
            if node.func == "NEG":
                return f"(-({arg}))"
            return f"{UNARY_CPP[node.func]}({arg})"
        if isinstance(node, FMA):
            a, b, c = self.expr(node.a), self.expr(node.b), self.expr(node.c)
            return f"fmaf({a}, {b}, {c})"
        if isinstance(node, Compare):
            left = self.expr(node.left)
            right = self.expr(node.right)
            return f"({left} {COMPARE_CPP[node.op]} {right})"
        if isinstance(node, Ternary):
            cond = self.expr(node.cond)
            t = self.expr(node.if_true)
            f = self.expr(node.if_false)
            return f"({cond} ? ({t}) : ({f}))"
        raise TypeError(f"Unknown AST node: {type(node)}")

    def kernel_source(self, root: ASTNode, output_var: str = "O") -> str:
        """Produce full __global__ kernel source. Single output: O[i] = expr;"""
        expr_str = self.expr(root)
        # Kernel params: A, B, C, X, Y, O, N
        params = "float* A, float* B, float* C, float* X, float* Y, float* O, int N"
        body = f"""
extern "C" __global__ void k({params}) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {{
        {output_var}[i] = {expr_str};
    }}
}}
""".strip()
        return body


def ast_to_cuda(ast_string: str, dtype: str = DEFAULT_DTYPE) -> str:
    """
    Convert S-expression AST string to full CUDA kernel source.
    Guarantees syntactically valid C++.
    """
    from ptx_decompiler.data.ast_nodes import parse_sexp

    tree = parse_sexp(ast_string)
    r = CUDARenderer(dtype=dtype)
    return r.kernel_source(tree)
