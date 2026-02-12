"""Evaluation metrics: exact match, tree edit distance, compilation success, semantic equivalence."""

from typing import List, Optional, Callable

import torch


def exact_match_accuracy(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    eos_id: int,
) -> float:
    """Exact match: predicted sequence (up to EOS) equals target. Batched."""
    B = predicted_ids.size(0)
    correct = 0
    for b in range(B):
        pred = predicted_ids[b]
        tgt = target_ids[b]
        pred_list = pred.tolist()
        tgt_list = tgt.tolist()
        if eos_id in pred_list:
            pred_list = pred_list[: pred_list.index(eos_id) + 1]
        if eos_id in tgt_list:
            tgt_list = tgt_list[: tgt_list.index(eos_id) + 1]
        pred_trim = [x for x in pred_list if x != pad_id and x != eos_id]
        tgt_trim = [x for x in tgt_list if x != pad_id and x != eos_id]
        if pred_trim == tgt_trim:
            correct += 1
    return correct / B if B else 0.0


def _levenshtein(s1: List[int], s2: List[int]) -> int:
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def compute_tree_edit_distance(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    eos_id: int,
) -> float:
    """Normalized token-level edit distance (1 - dist/max_len). Batched, return average."""
    B = predicted_ids.size(0)
    total_sim = 0.0
    for b in range(B):
        pred = predicted_ids[b].tolist()
        tgt = target_ids[b].tolist()
        if eos_id in pred:
            pred = pred[: pred.index(eos_id)]
        if eos_id in tgt:
            tgt = tgt[: tgt.index(eos_id)]
        pred = [x for x in pred if x != pad_id]
        tgt = [x for x in tgt if x != pad_id]
        dist = _levenshtein(pred, tgt)
        max_len = max(len(pred), len(tgt), 1)
        total_sim += 1.0 - (dist / max_len)
    return total_sim / B if B else 0.0


def compile_success_rate(
    ast_strings: List[str],
    render_fn: Callable[[str], str],
    compile_fn: Callable[[str], bool],
) -> float:
    """Fraction of ASTs that render to CUDA and compile successfully."""
    if not ast_strings:
        return 0.0
    ok = 0
    for ast in ast_strings:
        try:
            cuda = render_fn(ast)
            if compile_fn(cuda):
                ok += 1
        except Exception:
            pass
    return ok / len(ast_strings)


def semantic_equivalence_rate(
    original_cuda: List[str],
    decompiled_cuda: List[str],
    run_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """Fraction of pairs where original and decompiled kernels produce same output. Optional run_fn."""
    if run_fn is None or not original_cuda or not decompiled_cuda:
        return 0.0
    ok = 0
    for o, d in zip(original_cuda, decompiled_cuda):
        try:
            if run_fn(o, d):
                ok += 1
        except Exception:
            pass
    return ok / len(original_cuda)
