"""Training loop, metrics, curriculum, and LR scheduler."""

from ptx_decompiler.training.trainer import Trainer
from ptx_decompiler.training.metrics import (
    exact_match_accuracy,
    compute_tree_edit_distance,
    compile_success_rate,
)
from ptx_decompiler.training.curriculum import get_max_tier_for_epoch
from ptx_decompiler.training.scheduler import get_cosine_schedule_with_warmup

__all__ = [
    "Trainer",
    "exact_match_accuracy",
    "compute_tree_edit_distance",
    "compile_success_rate",
    "get_max_tier_for_epoch",
    "get_cosine_schedule_with_warmup",
]
