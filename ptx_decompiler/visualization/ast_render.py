"""Render AST S-expression as Graphviz tree (SVG string)."""

from typing import Optional

from ptx_decompiler.data.ast_nodes import parse_sexp, ASTNode, BinOp, UnaryFunc, FMA, Compare, Ternary, Var, Literal


def _node_label(node: ASTNode) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Literal):
        return str(node.value)
    if isinstance(node, BinOp):
        return node.op
    if isinstance(node, UnaryFunc):
        return node.func
    if isinstance(node, FMA):
        return "FMA"
    if isinstance(node, Compare):
        return node.op
    if isinstance(node, Ternary):
        return "?"
    return ""


def _add_edges(dot_lines: list, node: ASTNode, parent_id: Optional[str], node_id: int) -> int:
    label = _node_label(node)
    curr_id = f"n{node_id}"
    dot_lines.append(f'  {curr_id} [label="{label}"];')
    if parent_id is not None:
        dot_lines.append(f"  {parent_id} -> {curr_id};")
    next_id = node_id + 1

    if isinstance(node, (BinOp, Compare)):
        next_id = _add_edges(dot_lines, node.left, curr_id, next_id)
        next_id = _add_edges(dot_lines, node.right, curr_id, next_id)
    elif isinstance(node, UnaryFunc):
        next_id = _add_edges(dot_lines, node.arg, curr_id, next_id)
    elif isinstance(node, FMA):
        next_id = _add_edges(dot_lines, node.a, curr_id, next_id)
        next_id = _add_edges(dot_lines, node.b, curr_id, next_id)
        next_id = _add_edges(dot_lines, node.c, curr_id, next_id)
    elif isinstance(node, Ternary):
        next_id = _add_edges(dot_lines, node.cond, curr_id, next_id)
        next_id = _add_edges(dot_lines, node.if_true, curr_id, next_id)
        next_id = _add_edges(dot_lines, node.if_false, curr_id, next_id)
    return next_id


def ast_to_graphviz_svg(ast_sexp: str) -> str:
    """Return SVG string of the AST tree. Requires graphviz system package or pip install graphviz."""
    try:
        from graphviz import Source
    except ImportError:
        return "<p>Install graphviz: pip install graphviz</p>"

    tree = parse_sexp(ast_sexp)
    dot_lines = ["digraph G {", '  node [shape=circle, fontsize=10];']
    _add_edges(dot_lines, tree, None, 0)
    dot_lines.append("}")
    dot_src = "\n".join(dot_lines)
    src = Source(dot_src)
    return src.pipe(format="svg").decode("utf-8")
