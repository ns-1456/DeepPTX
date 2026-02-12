"""Verify CUDA source compiles with nvcc."""

from typing import Optional

from ptx_decompiler.data.compiler import compile_cuda_to_ptx


class CompilationVerifier:
    def __init__(self, nvcc_path: str = "nvcc", arch: str = "sm_75"):
        self.nvcc_path = nvcc_path
        self.arch = arch

    def check(self, cuda_source: str, work_dir: Optional[str] = None) -> bool:
        ok, _ = compile_cuda_to_ptx(
            cuda_source,
            work_dir=work_dir,
            nvcc_path=self.nvcc_path,
            arch=self.arch,
        )
        return ok
