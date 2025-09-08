"""PyTorch Dataset and DataLoader for (PTX, AST) pairs with padding and curriculum sampling."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler

from ptx_decompiler.tokenizer import PTXTokenizer, ASTTokenizer


class PTXASTDataset(Dataset):
    """Dataset of (ptx_ids, ast_ids) with optional tier for curriculum."""

    def __init__(
        self,
        ptx_strings: List[str],
        ast_strings: List[str],
        ptx_tokenizer: PTXTokenizer,
        ast_tokenizer: ASTTokenizer,
        tiers: Optional[List[int]] = None,
        max_ptx_len: Optional[int] = None,
        max_ast_len: Optional[int] = None,
    ):
        assert len(ptx_strings) == len(ast_strings)
        self.ptx_strings = ptx_strings
        self.ast_strings = ast_strings
        self.ptx_tokenizer = ptx_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.tiers = tiers or [1] * len(ptx_strings)
        self.max_ptx_len = max_ptx_len
        self.max_ast_len = max_ast_len

    def __len__(self) -> int:
        return len(self.ptx_strings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ptx_ids = self.ptx_tokenizer.encode(self.ptx_strings[idx], add_bos_eos=True)
        ast_ids = self.ast_tokenizer.encode(self.ast_strings[idx], add_bos_eos=True)
        return {
            "ptx_ids": torch.tensor(ptx_ids, dtype=torch.long),
            "ast_ids": torch.tensor(ast_ids, dtype=torch.long),
            "tier": self.tiers[idx],
        }


def collate_pad_batch(
    batch: List[Dict[str, Any]],
    pad_id_ptx: int,
    pad_id_ast: int,
    max_ptx_len: Optional[int] = None,
    max_ast_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad sequences to max length in batch (or global max) and stack."""
    ptx_list = [b["ptx_ids"] for b in batch]
    ast_list = [b["ast_ids"] for b in batch]
    ptx_lens = [len(p) for p in ptx_list]
    ast_lens = [len(a) for a in ast_list]
    max_ptx = max_ptx_len or max(ptx_lens)
    max_ast = max_ast_len or max(ast_lens)

    ptx_padded = torch.full((len(batch), max_ptx), pad_id_ptx, dtype=torch.long)
    ast_padded = torch.full((len(batch), max_ast), pad_id_ast, dtype=torch.long)
    for i, (p, a) in enumerate(zip(ptx_list, ast_list)):
        ptx_padded[i, : len(p)] = p
        ast_padded[i, : len(a)] = a

    # For decoder: input is ast[:-1], target is ast[1:]
    ast_input = ast_padded[:, :-1]
    ast_target = ast_padded[:, 1:]

    return {
        "ptx_ids": ptx_padded,
        "ptx_mask": ptx_padded != pad_id_ptx,
        "ast_input_ids": ast_input,
        "ast_target_ids": ast_target,
        "ast_mask": ast_target != pad_id_ast,
        "tiers": torch.tensor([b["tier"] for b in batch], dtype=torch.long),
    }


class CurriculumSampler(Sampler[int]):
    """Samples indices so that early epochs see only low-tier examples (tier <= max_tier)."""

    def __init__(
        self,
        dataset: PTXASTDataset,
        epoch: int = 0,
        max_tier_by_epoch: Optional[Callable[[int], int]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.epoch = epoch
        self.shuffle = shuffle
        self.seed = seed
        # Default: epochs 0-4 tier 2, 5-9 tier 3, 10-14 tier 4, 15+ all
        self.max_tier_by_epoch = max_tier_by_epoch or (
            lambda e: 2 if e < 5 else (3 if e < 10 else (4 if e < 15 else 7))
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        max_tier = self.max_tier_by_epoch(self.epoch)
        indices = [
            i
            for i in range(len(self.dataset))
            if self.dataset.tiers[i] <= max_tier
        ]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g)
            indices = [indices[i] for i in perm]
        return iter(indices)

    def __len__(self) -> int:
        max_tier = self.max_tier_by_epoch(self.epoch)
        return sum(1 for i in range(len(self.dataset)) if self.dataset.tiers[i] <= max_tier)


def load_parquet_for_training(
    parquet_path: Union[str, Path],
    ptx_tokenizer: PTXTokenizer,
    ast_tokenizer: ASTTokenizer,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[PTXASTDataset, PTXASTDataset]:
    """Load Parquet dataset and split into train/val. Tokenizers must already be fitted."""
    import pandas as pd

    path = Path(parquet_path)
    df = pd.read_parquet(path)
    n = len(df)
    torch.manual_seed(seed)
    perm = torch.randperm(n)
    split = int(n * train_ratio)
    train_idx = perm[:split].tolist()
    val_idx = perm[split:].tolist()

    train_ds = PTXASTDataset(
        ptx_strings=df["ptx_normalized"].iloc[train_idx].tolist(),
        ast_strings=df["ast_sexp"].iloc[train_idx].tolist(),
        ptx_tokenizer=ptx_tokenizer,
        ast_tokenizer=ast_tokenizer,
        tiers=df["tier"].iloc[train_idx].tolist(),
    )
    val_ds = PTXASTDataset(
        ptx_strings=df["ptx_normalized"].iloc[val_idx].tolist(),
        ast_strings=df["ast_sexp"].iloc[val_idx].tolist(),
        ptx_tokenizer=ptx_tokenizer,
        ast_tokenizer=ast_tokenizer,
        tiers=df["tier"].iloc[val_idx].tolist(),
    )
    return train_ds, val_ds
