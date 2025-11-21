import platform
import sys

from importlib.metadata import version


def print_env_report():
    """

    Prints a report of the environment.  Useful for debugging and reproducibility.
    Usage:
    ```
    python -m liger_kernel.env_report
    ```

    """
    print("Environment Report:")
    print("-------------------")
    print(f"Operating System: {platform.platform()}")
    print(f"Python version: {sys.version.split()[0]}")

    try:
        print(f"Liger Kernel version: {version('liger-kernel')}")
    except ImportError:
        print("Liger Kernel: Not installed")

    try:
        import torch
        import os

        from liger_kernel.utils import is_cuda_available, is_npu_available

        print(f"PyTorch version: {torch.__version__}")
        cuda_version = torch.version.cuda if is_cuda_available() else "Not available"
        print(f"CUDA version: {cuda_version}")
        hip_version = torch.version.hip if is_cuda_available() and torch.version.hip else "Not available"
        print(f"HIP(ROCm) version: {hip_version}")

        # Ascend NPU
        npu_available = is_npu_available()
        print(f"Ascend NPU available: {npu_available}")
        if npu_available:
            ascend_id = os.environ.get("ASCEND_DEVICE_ID", "")
            npu_vis = os.environ.get("NPU_VISIBLE_DEVICES", "")
            if ascend_id:
                print(f"ASCEND_DEVICE_ID: {ascend_id}")
            if npu_vis:
                print(f"NPU_VISIBLE_DEVICES: {npu_vis}")

    except ImportError:
        print("PyTorch: Not installed")
        print("CUDA version: Unable to query")
        print("HIP(ROCm) version: Unable to query")

    try:
        import triton

        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: Not installed")

    try:
        import transformers

        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

    try:
        xpu_version = torch.version.xpu if torch.xpu.is_available() else "XPU Not Available"
        print(f"XPU version: {xpu_version}")
    except ImportError:
        print("XPU version: Unable to query")


if __name__ == "__main__":
    print_env_report()
