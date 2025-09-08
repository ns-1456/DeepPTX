"""Curriculum learning: max tier allowed per epoch."""

def get_max_tier_for_epoch(epoch: int) -> int:
    """
    Epochs 0-4: tier 2 (simple + nested)
    Epochs 5-9: tier 3 (+ unary/FMA)
    Epochs 10-14: tier 4 (+ ternary)
    Epochs 15-19: tier 5 (+ type diversity)
    Epochs 20+: tier 7 (all)
    """
    if epoch < 5:
        return 2
    if epoch < 10:
        return 3
    if epoch < 15:
        return 4
    if epoch < 20:
        return 5
    return 7
