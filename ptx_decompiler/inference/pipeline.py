"""End-to-end PTX -> AST -> CUDA decompilation pipeline."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from ptx_decompiler.data import normalize_ptx, parse_sexp
from ptx_decompiler.data.renderer import CUDARenderer
from ptx_decompiler.inference.beam_search import tree_constrained_beam_decode
from ptx_decompiler.inference.verifier import CompilationVerifier

if TYPE_CHECKING:
    from ptx_decompiler.tokenizer import PTXTokenizer, ASTTokenizer


@dataclass
class DecompilationResult:
    ptx_input: str
    ast: str
    cuda: str
    compiles: bool
    attention_weights: Optional[torch.Tensor] = None
    confidence: Optional[float] = None


class DecompilationPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        ptx_tokenizer: "PTXTokenizer",
        ast_tokenizer: "ASTTokenizer",
        device: Optional[torch.device] = None,
        verifier: Optional[CompilationVerifier] = None,
        beam_size: int = 4,
    ):
        self.model = model
        self.ptx_tokenizer = ptx_tokenizer
        self.ast_tokenizer = ast_tokenizer
        from ptx_decompiler.utils import get_device
        self.device = device or get_device()
        self.verifier = verifier or CompilationVerifier()
        self.beam_size = beam_size
        self.renderer = CUDARenderer()
        self._last_attention: Optional[torch.Tensor] = None

    def decompile(self, raw_ptx: str) -> DecompilationResult:
        self.model.to(self.device)
        self.model.eval()
        ptx_clean = normalize_ptx(raw_ptx)
        ptx_ids = self.ptx_tokenizer.encode(ptx_clean, add_bos_eos=True)
        ptx_t = torch.tensor([ptx_ids], dtype=torch.long, device=self.device)
        ast_ids, attn = tree_constrained_beam_decode(
            self.model,
            ptx_t.squeeze(0),
            None,
            self.ast_tokenizer,
            self.device,
            beam_size=self.beam_size,
        )
        self._last_attention = attn
        ast_sexp = self.ast_tokenizer.decode(ast_ids, skip_special=True)
        try:
            tree = parse_sexp(ast_sexp)
            cuda_code = self.renderer.kernel_source(tree)
        except Exception:
            cuda_code = ""
        compiles = self.verifier.check(cuda_code) if cuda_code else False
        return DecompilationResult(
            ptx_input=raw_ptx,
            ast=ast_sexp,
            cuda=cuda_code,
            compiles=compiles,
            attention_weights=attn,
            confidence=None,
        )


