"""Compile CUDA source to PTX using nvcc. Supports concurrent use with unique filenames."""

import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional, Tuple

# Thread-local counter for unique filenames in concurrent compilation
_local = threading.local()


def _unique_name() -> str:
    """Return a short unique name for temp files (thread-safe)."""
    if not hasattr(_local, "counter"):
        _local.counter = 0
    _local.counter += 1
    return f"{os.getpid()}_{threading.get_ident()}_{_local.counter}"


def compile_cuda_to_ptx(
    cuda_source: str,
    work_dir: Optional[str] = None,
    nvcc_path: str = "nvcc",
    arch: str = "sm_75",
    opt_level: str = "-O0",
) -> Tuple[bool, str]:
    """
    Compile CUDA source to PTX.
    Thread/process-safe: uses unique filenames per call.
    Use opt_level="-O0" for fast generation, "-O3" for realistic optimized PTX.

    Returns:
        (success, ptx_content_or_error_message)
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="ptx_compile_")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = _unique_name()
    cu_path = work_dir / f"{tag}.cu"
    ptx_path = work_dir / f"{tag}.ptx"

    try:
        cu_path.write_text(cuda_source, encoding="utf-8")
        result = subprocess.run(
            [nvcc_path, "-ptx", opt_level, f"-arch={arch}", str(cu_path), "-o", str(ptx_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(work_dir),
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout or "nvcc failed"
        if not ptx_path.exists():
            return False, "PTX file was not produced"
        ptx_text = ptx_path.read_text(encoding="utf-8")
        # Clean up temp files immediately to avoid disk bloat
        try:
            cu_path.unlink(missing_ok=True)
            ptx_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True, ptx_text
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
