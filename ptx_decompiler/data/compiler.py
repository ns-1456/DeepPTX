"""Compile CUDA source to PTX using nvcc."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def compile_cuda_to_ptx(
    cuda_source: str,
    work_dir: Optional[str] = None,
    nvcc_path: str = "nvcc",
    arch: str = "sm_75",
) -> Tuple[bool, str]:
    """
    Compile CUDA source to PTX.

    Returns:
        (success, ptx_content_or_error_message)
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="ptx_compile_")
    work_dir = Path(work_dir)
    cu_path = work_dir / "temp.cu"
    ptx_path = work_dir / "temp.ptx"

    try:
        cu_path.write_text(cuda_source, encoding="utf-8")
        result = subprocess.run(
            [nvcc_path, "-ptx", "-O3", f"-arch={arch}", str(cu_path), "-o", str(ptx_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(work_dir),
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout or "nvcc failed"
        if not ptx_path.exists():
            return False, "PTX file was not produced"
        return True, ptx_path.read_text(encoding="utf-8")
    except subprocess.TimeoutExpired:
        return False, "nvcc timed out"
    except FileNotFoundError:
        return False, "nvcc not found (CUDA toolkit not installed?)"
    except Exception as e:
        return False, str(e)


def compile_cuda_to_ptx_silent(cuda_source: str, work_dir: Optional[str] = None) -> Optional[str]:
    """
    Compile CUDA to PTX; on success return PTX string, on failure return None.
    Suppresses stderr from nvcc.
    """
    ok, out = compile_cuda_to_ptx(cuda_source, work_dir=work_dir)
    return out if ok else None
