"""Word-level PTX tokenizer. Vocabulary built from training PTX sequences."""

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

# Special tokens
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def _tokenize_ptx_line(line: str) -> List[str]:
    """Split a line into tokens: keep mnemonics, registers, punctuation as separate tokens."""
    tokens: List[str] = []
    # Split on whitespace
    parts = line.split()
    for p in parts:
        p = p.rstrip(";")
        if "," in p:
            for i, sub in enumerate(p.split(",")):
                if i > 0:
                    tokens.append(",")
                tokens.append(sub.strip())
        else:
            tokens.append(p)
    return tokens


def tokenize_ptx_text(ptx_normalized: str) -> List[str]:
    """Tokenize normalized PTX string (space-separated instructions) into word-level tokens."""
    tokens: List[str] = []
    for part in ptx_normalized.split():
        if part in (",", ";", "[", "]", "(", ")"):
            tokens.append(part)
        else:
            tokens.append(part)
    return tokens


class PTXTokenizer:
    """Word-level tokenizer for PTX. Build vocab from a list of PTX strings."""

    def __init__(
        self,
        vocab: Optional[dict] = None,
        max_vocab_size: int = 2000,
        min_frequency: int = 1,
    ):
        self.vocab = vocab or {}
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self._idx2token: List[str] = []
        if self.vocab:
            self._idx2token = [""] * (max(self.vocab.values()) + 1)
            for t, i in self.vocab.items():
                self._idx2token[i] = t

    def build_vocab(self, ptx_strings: List[str]) -> None:
        """Build vocabulary from a list of normalized PTX strings."""
        counter: Counter = Counter()
        for s in ptx_strings:
            for t in tokenize_ptx_text(s):
                counter[t] += 1

        self.vocab = {}
        for tok in SPECIAL_TOKENS:
            self.vocab[tok] = len(self.vocab)
        for tok, count in counter.most_common(self.max_vocab_size - len(SPECIAL_TOKENS)):
            if count < self.min_frequency:
                break
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                if len(self.vocab) >= self.max_vocab_size:
                    break

        self._idx2token = [""] * len(self.vocab)
        for t, i in self.vocab.items():
            self._idx2token[i] = t

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        ptx_normalized: str,
        add_bos_eos: bool = True,
    ) -> List[int]:
        """Encode normalized PTX string to list of token ids."""
        tokens = tokenize_ptx_text(ptx_normalized)
        ids = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for t in tokens:
            ids.append(self.vocab.get(t, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """Decode list of token ids back to normalized PTX string."""
        tokens = []
        for i in ids:
            if i < 0 or i >= len(self._idx2token):
                continue
            t = self._idx2token[i]
            if skip_special and t in SPECIAL_TOKENS:
                if t == EOS:
                    break
                continue
            tokens.append(t)
        return " ".join(tokens)

    def save(self, path: Union[str, Path]) -> None:
        """Save vocab to file (one token per line, index implied by line order)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(self._idx2token)):
                f.write(self._idx2token[i] + "\n")

    def load(self, path: Union[str, Path]) -> None:
        """Load vocab from file."""
        path = Path(path)
        self._idx2token = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self._idx2token.append(line.rstrip("\n"))
        self.vocab = {t: i for i, t in enumerate(self._idx2token)}
