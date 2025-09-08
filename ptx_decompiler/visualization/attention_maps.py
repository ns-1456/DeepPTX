"""Cross-attention heatmap: PTX tokens (x) vs AST tokens (y)."""

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(
    attention: np.ndarray,
    ptx_tokens: List[str],
    ast_tokens: List[str],
    title: str = "Cross-Attention (PTX → AST)",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    attention: (T_ast, S_ptx) or (n_heads, T_ast, S_ptx) - will average over heads if 3D.
    """
    if attention.ndim == 3:
        attention = attention.mean(axis=0)
    h, w = attention.shape
    if len(ast_tokens) != h:
        ast_tokens = ast_tokens[:h] or [str(i) for i in range(h)]
    if len(ptx_tokens) != w:
        ptx_tokens = ptx_tokens[:w] or [str(j) for j in range(w)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticklabels(ptx_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ast_tokens, fontsize=8)
    ax.set_xlabel("PTX tokens")
    ax.set_ylabel("AST tokens")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
