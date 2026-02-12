"""PTX cleaner and canonicalizer. Strips directives, normalizes registers and formatting."""

import re
from typing import Dict, List, Tuple


# Directives to strip (we keep only instruction body)
STRIP_DIRECTIVES = (
    ".version",
    ".target",
    ".address_size",
    ".visible",
    ".entry",
    ".loc",
    ".file",
    ".section",
    ".align",
    ".reg",
    ".param",
    ".global",
    ".local",
    ".shared",
    ".func",
    ".callprototype",
    ".calltarget",
)


def _is_directive_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("//"):
        return True
    for d in STRIP_DIRECTIVES:
        if stripped.startswith(d + " ") or stripped.startswith(d + "\t") or stripped == d:
            return True
    # .something that looks like a directive
    if stripped.startswith(".") and not stripped.startswith(".visible"):
        return True
    return False


def _extract_instructions(lines: List[str]) -> List[str]:
    """Keep only instruction lines (no directives/comments)."""
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if _is_directive_line(line):
            continue
        out.append(s)
    return out


def _tokenize_ptx_line(line: str) -> List[str]:
    """Rough tokenization: split on whitespace and punctuation, keep tokens."""
    # Split on whitespace first
    parts = line.split()
    tokens: List[str] = []
    for p in parts:
        # Split trailing ; and ,
        p = p.rstrip(";")
        if "," in p:
            for i, sub in enumerate(p.split(",")):
                if i > 0:
                    tokens.append(",")
                tokens.append(sub.strip())
        else:
            tokens.append(p)
    return tokens


def _canonicalize_registers(instructions: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Rename registers to canonical %f0, %f1, ... and %r0, %r1, ... (by prefix and order).
    Returns (new_instruction_lines, old_to_new_map).
    """
    reg_pattern = re.compile(r"%[a-zA-Z_][a-zA-Z0-9_]*")
    float_regs: List[str] = []
    int_regs: List[str] = []
    other_regs: List[str] = []
    seen: set = set()

    for line in instructions:
        for m in reg_pattern.finditer(line):
            name = m.group(0)
            if name in seen:
                continue
            seen.add(name)
            if name.startswith("%f"):
                float_regs.append(name)
            elif name.startswith("%r"):
                int_regs.append(name)
            else:
                other_regs.append(name)

    mapping: Dict[str, str] = {}
    for i, r in enumerate(float_regs):
        mapping[r] = f"%f{i}"
    for i, r in enumerate(int_regs):
        mapping[r] = f"%r{i}"
    for i, r in enumerate(other_regs):
        mapping[r] = f"%s{i}"

    def replace_line(line: str) -> str:
        for old, new in mapping.items():
            line = line.replace(old, new)
        return line

    new_lines = [replace_line(l) for l in instructions]
    return new_lines, mapping


def normalize_ptx(raw_ptx: str) -> str:
    """
    Clean and canonicalize PTX:
    - Remove comments and directive lines
    - Keep only instruction body
    - Canonicalize register names to %f0, %f1, %r0, ...
    - Collapse whitespace, single space between tokens
    """
    lines = raw_ptx.splitlines()
    instructions = _extract_instructions(lines)
    if not instructions:
        return ""

    canonical, _ = _canonicalize_registers(instructions)
    # Normalize spacing: one space between tokens
    normalized_lines = []
    for line in canonical:
        tokens = _tokenize_ptx_line(line)
        normalized_lines.append(" ".join(tokens))

    return " ".join(normalized_lines)
