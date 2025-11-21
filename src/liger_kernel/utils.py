try:
    import peft  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import torch


def is_peft_available():
    return PEFT_AVAILABLE


def infer_device():
    """
    Get current device name based on available devices
    """
    # Prefer Ascend NPU if available (torch.npu)
    if is_npu_available():
        return "npu"

    # CUDA (NVIDIA / AMD) if available
    if is_cuda_available():
        return "cuda"

    # XPU (Intel) if available
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return "xpu"

    return "cpu"


def is_npu_available() -> bool:
    """Detect Ascend NPU availability.

    Checks `torch.npu.is_available()` if present, and falls back to common
    Ascend environment variables such as `ASCEND_DEVICE_ID` or
    `NPU_VISIBLE_DEVICES`.
    """
    import os

    try:
        if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
            return True
    except Exception:
        # Defensive: if torch.npu exists but calling fails, ignore.
        pass

    # Common Ascend runtime environment variables
    if os.environ.get("ASCEND_DEVICE_ID") is not None:
        return True
    if os.environ.get("NPU_VISIBLE_DEVICES") is not None:
        return True

    # No reliable programmatic check available; assume not present.
    return False


def is_cuda_available() -> bool:
    """Wrapper around torch.cuda availability check."""
    try:
        return getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    except Exception:
        return False


def get_device(prefer: str | None = None) -> str:
    """Return a preferred device name as string: 'npu', 'cuda', 'xpu', or 'cpu'.

    If `prefer` is provided it will try that device first.
    """
    if prefer == "npu" and is_npu_available():
        return "npu"
    if prefer == "cuda" and is_cuda_available():
        return "cuda"

    # fallback ordering: npu -> cuda -> xpu -> cpu
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return "xpu"
    return "cpu"


def get_torch_device(device: str | None = None) -> torch.device:
    """Return a `torch.device` corresponding to a device string.

    For NPU, attempts `torch.device('npu')` but falls back to CPU if not supported
    by the local torch build.
    """
    dev = device or get_device()
    if dev == "cuda":
        return torch.device("cuda")
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "xpu":
        try:
            return torch.device("xpu")
        except Exception:
            return torch.device("cpu")
    if dev == "npu":
        try:
            return torch.device("npu")
        except Exception:
            # Some torch builds may not recognize 'npu' as a device string; fall
            # back to CPU to keep behavior safe.
            return torch.device("cpu")
    # Unknown device, default to CPU
    return torch.device("cpu")


def to_device(obj, device: str | None = None):
    """Move tensor/module-like `obj` to `device` when possible.

    This is a best-effort helper that will silently return the original
    object if the move fails.
    """
    try:
        torch_dev = get_torch_device(device)
        if hasattr(obj, "to"):
            return obj.to(torch_dev)
    except Exception:
        pass
    return obj


def get_multi_processor_count(device) -> int:
    """Return a heuristic multi-processor count for `device`.

    For CUDA devices this wraps `torch.cuda.get_device_properties(...).multi_processor_count`.
    For XPU/NPU it attempts to query analogous properties where available, and
    falls back to 1 if nothing can be queried.
    """
    try:
        # CUDA
        if is_cuda_available():
            try:
                return torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                pass

        # XPU
        if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
            try:
                return torch.xpu.get_device_properties(device).gpu_eu_count
            except Exception:
                pass

        # NPU (Ascend) - best-effort
        if is_npu_available():
            try:
                if hasattr(torch, "npu") and hasattr(torch.npu, "get_device_properties"):
                    return torch.npu.get_device_properties(device).multi_processor_count
            except Exception:
                pass
    except Exception:
        pass

    # Reasonable default to avoid division by zero
    return 1


def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        return before_fn(*before_args, **before_kwargs)
    else:
        return after_fn(*after_args, **after_kwargs)
