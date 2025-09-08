"""AST S-expression tokenizer. Closed vocabulary for operators and variables."""

from pathlib import Path
from typing import List, Union

from ptx_decompiler.tokenizer.ptx_tokenizer import PAD, BOS, EOS, SPECIAL_TOKENS

# AST-specific tokens: parentheses, operators, variables
AST_OPERATORS = [
    "ADD", "SUB", "MUL", "DIV", "MAX", "MIN",
    "SIN", "COS", "EXP", "LOG", "SQRT", "RSQRT", "NEG", "ABS",
    "FMA", "TERNARY",
    "GT", "LT", "GE", "LE", "EQ", "NE",
]
AST_VARS = ["A", "B", "C", "X", "Y", "O", "T"]
AST_LITERALS = ["0.0", "1.0", "-1.0", "0.5", "2.0"]
AST_EXTRA = ["(", ")"]

# Full closed vocab for AST (UNK for any literal/var not in list)
AST_VOCAB_LIST = list(SPECIAL_TOKENS) + ["("] + [")"] + AST_OPERATORS + AST_VARS + AST_LITERALS


def tokenize_ast_sexp(sexp: str) -> List[str]:
    """Tokenize S-expression string at word level. (ADD (MUL A B) C) -> ['(', 'ADD', '(', 'MUL', 'A', 'B', ')', 'C', ')']."""
    tokens: List[str] = []
    i = 0
    s = sexp.strip()
    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        if s[i] == "(":
            tokens.append("(")
            i += 1
        elif s[i] == ")":
            tokens.append(")")
            i += 1
        else:
            start = i
            while i < len(s) and not s[i].isspace() and s[i] not in "()":
                i += 1
            if start < i:
                tokens.append(s[start:i])
    return tokens


class ASTTokenizer:
    """Tokenizer for AST S-expressions. Fixed closed vocabulary."""

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(AST_VOCAB_LIST)}
        self._idx2token = AST_VOCAB_LIST.copy()

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        ast_sexp: str,
        add_bos_eos: bool = True,
    ) -> List[int]:
        """Encode S-expression string to list of token ids. Unknown tokens map to PAD (closed vocab)."""
        tokens = tokenize_ast_sexp(ast_sexp)
        ids = []
        if add_bos_eos:
            ids.append(self.bos_id)
        unk_id = self.vocab.get("<UNK>", self.pad_id)
        for t in tokens:
            ids.append(self.vocab.get(t, unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """Decode list of token ids to S-expression string."""
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
        """Save vocab (same format as PTX: one token per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for t in self._idx2token:
                f.write(t + "\n")

    def load(self, path: Union[str, Path]) -> None:
        """Load vocab from file."""
        path = Path(path)
        self._idx2token = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self._idx2token.append(line.rstrip("\n"))
        self.vocab = {t: i for i, t in enumerate(self._idx2token)}
