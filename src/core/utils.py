"""Shared utilities for Zero-3DCE: resolution alignment and normalization."""
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def pad_to_align(tensor: torch.Tensor, align: int = 8) -> tuple[torch.Tensor, int, int]:
    """Pad a (..., H, W) tensor's spatial dims to the nearest multiple of `align`.
    Uses reflect padding to avoid introducing hard edges at the boundary.
    Returns (padded_tensor, pad_h, pad_w) so the caller can unpad after inference."""
    H, W  = tensor.shape[-2], tensor.shape[-1]
    pad_h = (-H) % align
    pad_w = (-W) % align
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def unpad(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Strip padding added by pad_to_align from the last two dimensions."""
    if pad_h:
        tensor = tensor[..., :-pad_h, :]
    if pad_w:
        tensor = tensor[..., :, :-pad_w]
    return tensor


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Normalize a (B, 3, ...) float32 [0, 1] tensor to ImageNet mean/std.
    Works for any number of trailing spatial dimensions (2D or 5D)."""
    ndim  = x.dim() - 2                                       # number of spatial dims
    shape = (1, 3) + (1,) * ndim
    mean  = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(shape)
    std   = torch.tensor(_IMAGENET_STD,  device=x.device, dtype=x.dtype).view(shape)
    return (x - mean) / std
