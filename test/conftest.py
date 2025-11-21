import pytest
import torch

from liger_kernel.utils import is_cuda_available, is_npu_available


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if is_cuda_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif is_npu_available():
        # Try best-effort to clear NPU cache if available
        try:
            if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
        except Exception:
            pass
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
