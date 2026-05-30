"""
Training-only optical flow module using frozen RAFT-small.

NOT imported by inference.py, camera_demo.py, or trt_infer.py — zero
inference cost.  Requires torchvision >= 0.13.

Public API
----------
bilinear_warp(frame, flow)          -> warped frame
FrozenRAFT                          -> frozen RAFT-small wrapper
WarpTemporalLoss                    -> flow-compensated temporal consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bilinear warp (pure PyTorch, no RAFT dependency)
# ---------------------------------------------------------------------------
def bilinear_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp `frame` using `flow`.

    Args:
        frame: (B, C, H, W)  source frame in [0, 1]
        flow:  (B, 2, H, W)  optical flow in pixel units.
               flow[:, 0] = x-displacement (horizontal)
               flow[:, 1] = y-displacement (vertical)
               Convention: frame[x + dx, y + dy] ≈ target[x, y]

    Returns:
        (B, C, H, W)  warped frame; border pixels are clamped.
    """
    B, _, H, W = frame.shape
    dtype, device = frame.dtype, frame.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij',
    )
    base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    sample = base + flow                                        # (B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    sample[:, 0] = 2.0 * sample[:, 0] / max(W - 1, 1) - 1.0
    sample[:, 1] = 2.0 * sample[:, 1] / max(H - 1, 1) - 1.0
    sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 2)

    return F.grid_sample(frame, sample,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=True)


# ---------------------------------------------------------------------------
# Frozen RAFT-small wrapper
# ---------------------------------------------------------------------------
class FrozenRAFT(nn.Module):
    """
    Frozen RAFT-small for training-only optical flow estimation.

    All parameters are permanently frozen; the model is always in eval mode.
    Runs in FP32 regardless of the outer autocast context.

    Requires: torchvision >= 0.13
              pip install --upgrade torchvision
    """

    def __init__(self, num_flow_updates: int = 12):
        super().__init__()
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FrozenRAFT requires torchvision >= 0.13 with optical_flow support.\n"
                "  pip install --upgrade torchvision\n"
                f"  Original error: {exc}"
            ) from exc

        for p in self.raft.parameters():
            p.requires_grad_(False)
        self.num_flow_updates = num_flow_updates

    def train(self, mode: bool = True):
        # Always stay frozen / eval — override parent train()
        return super().train(False)

    @torch.no_grad()
    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        """
        Estimate forward optical flow from frame0 to frame1.

        Args:
            frame0, frame1: (B, 3, H, W) float in [0, 1].
                            H and W must be multiples of 8.

        Returns:
            flow: (B, 2, H, W) in pixel units (dx, dy) such that
                  frame0[x + dx, y + dy] ≈ frame1[x, y]
        """
        # RAFT expects [0, 255] float; run in FP32 to avoid AMP precision issues
        with torch.amp.autocast("cuda", enabled=False):
            f0 = frame0.float() * 255.0
            f1 = frame1.float() * 255.0
            flow_list = self.raft(f0, f1, num_flow_updates=self.num_flow_updates)
        return flow_list[-1].float()

    def estimate_consecutive_flows(
        self,
        low: torch.Tensor,
        use_gamma: bool = True,
    ) -> list[torch.Tensor]:
        """
        Estimate flows for all D-1 consecutive frame pairs in a clip.

        Args:
            low:       (B, 3, D, H, W) low-light input clip
            use_gamma: apply gamma=0.4 before estimation so RAFT sees
                       brighter content (recommended for dark inputs)

        Returns:
            List of D-1 flow tensors, each (B, 2, H, W).
            Empty list when D < 2.
        """
        D = low.shape[2]
        if D < 2:
            return []

        frames = low.pow(0.4) if use_gamma else low
        flows  = []
        for d in range(D - 1):
            f0   = frames[:, :, d,     :, :]
            f1   = frames[:, :, d + 1, :, :]
            flows.append(self(f0, f1))
        return flows


# ---------------------------------------------------------------------------
# Warp-compensated temporal consistency loss
# ---------------------------------------------------------------------------
class WarpTemporalLoss(nn.Module):
    """
    Flow-compensated temporal consistency loss.

    For each consecutive pair (t, t+1):
        L_t = || enhanced_{t+1} - warp(enhanced_t, flow_t) ||_1

    Averaged over all pairs and spatial locations.  Replaces the naive
    alpha-map temporal TV (IlluminationSmoothnessLoss.l_temp) when RAFT
    flow is available — prevents penalising legitimate motion.
    """

    def forward(
        self,
        enhanced: torch.Tensor,
        flows: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            enhanced: (B, 3, D, H, W)
            flows:    list of D-1 flow tensors, each (B, 2, H, W),
                      as returned by FrozenRAFT.estimate_consecutive_flows

        Returns:
            scalar loss tensor (differentiable through `enhanced`)
        """
        if not flows or enhanced.shape[2] < 2:
            return torch.tensor(0.0, device=enhanced.device)

        total = enhanced.new_zeros(1).squeeze()
        for d, flow in enumerate(flows):
            frame_t  = enhanced[:, :, d,     :, :]
            frame_t1 = enhanced[:, :, d + 1, :, :]
            warped   = bilinear_warp(frame_t, flow.detach())
            total    = total + (frame_t1 - warped).abs().mean()
        return total / len(flows)
