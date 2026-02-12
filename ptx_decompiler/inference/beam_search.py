"""Tree-constrained beam search for AST decoding."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _paren_balance(tokens: List[int], open_id: int, close_id: int) -> int:
    """Return balance: +1 for open, -1 for close. Positive means need more closes."""
    b = 0
    for t in tokens:
        if t == open_id:
            b += 1
        elif t == close_id:
            b -= 1
    return b


def tree_constrained_beam_decode(
    model: torch.nn.Module,
    ptx_ids: torch.Tensor,
    ptx_padding_mask: Optional[torch.Tensor],
    ast_tokenizer: "ASTTokenizer",
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 128,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    open_id: Optional[int] = None,
    close_id: Optional[int] = None,
) -> Tuple[List[int], Optional[torch.Tensor]]:
    """
    Decode AST token ids with beam search. Optionally prune beams that violate
    balanced parentheses (if open_id/close_id provided).
    Returns: best token id list, last attention weights (if available).
    """
    if open_id is None:
        open_id = ast_tokenizer.vocab.get("(", pad_id)
    if close_id is None:
        close_id = ast_tokenizer.vocab.get(")", pad_id)

    model.eval()
    B = 1
    src = ptx_ids.unsqueeze(0).to(device) if ptx_ids.dim() == 1 else ptx_ids.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if ptx_padding_mask is not None:
        pad_mask = ptx_padding_mask.unsqueeze(0).to(device) if ptx_padding_mask.dim() == 1 else ptx_padding_mask.to(device)
    else:
        pad_mask = (src == pad_id)

    beams: List[Tuple[List[int], float]] = [([bos_id], 0.0)]
    last_attn = None

    for step in range(max_len - 1):
        all_candidates: List[Tuple[List[int], float]] = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                all_candidates.append((seq, score))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(src, tgt, pad_mask)
            logits = logits[0, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
            for k in range(beam_size):
                new_token = topk_ids[k].item()
                new_score = score + topk_log_probs[k].item()
                new_seq = seq + [new_token]
                balance = _paren_balance(new_seq, open_id, close_id)
                if balance < 0:
                    continue
                if new_token == eos_id:
                    if balance != 0:
                        continue
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break
        ordered = sorted(all_candidates, key=lambda x: -x[1])
        beams = ordered[:beam_size]
        if all(seq[-1] == eos_id for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return best_seq, last_attn
