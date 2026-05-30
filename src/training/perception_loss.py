"""
Perception-aware training losses for Zero-3DCE Phase 1.

Uses a frozen pretrained SuperPoint backbone as a feature extractor.
Training-only — zero inference overhead at deployment.

Two loss terms share a single backbone instance:
  - SemanticTemporalLoss : feature-space consistency between consecutive enhanced frames
  - PerceptualLoss       : feature distance from enhanced output to a synthetic bright
                           reference (gamma-corrected input, no GT needed)

Combined via PerceptionLoss which wraps both.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.superpoint import SuperPoint


# ---------------------------------------------------------------------------
# Frozen SuperPoint Backbone
# ---------------------------------------------------------------------------
class _FrozenBackbone(nn.Module):
    """
    SuperPoint backbone for feature extraction.
    Outputs:
        - semi: [B, 65, H/8, W/8] (detector head)
        - desc: [B, 256, H/8, W/8] (descriptor head)
    
    All parameters are frozen and the module is always in eval mode.
    """
    def __init__(self):
        super().__init__()
        self.model = SuperPoint()
        
        # Load weights
        root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = root / "checkpoints" / "superpoint_v1.pth"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: SuperPoint weights not found at {ckpt_path}")
            
        for p in self.parameters():
            p.requires_grad_(False)

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 1, H, W) grayscale."""
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W) float in [0, 1]
        Returns: (semi, desc)
        """
        gray = self._to_gray(x)
        return self.model(gray)

    def train(self, mode: bool = True):
        return super().train(False)   # always eval


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------
class SemanticTemporalLoss(nn.Module):
    """
    Feature-space temporal consistency using SuperPoint.
    Penalises drift in keypoint distributions and descriptor maps.
    """
    def __init__(self, backbone: _FrozenBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        if enhanced.shape[2] < 2:
            return enhanced.new_zeros(())

        # (B, 3, D, H, W) -> frame0, frame1
        f0 = enhanced[:, :, 0]
        f1 = enhanced[:, :, 1]
        
        semi0, desc0 = self.backbone(f0)
        semi1, desc1 = self.backbone(f1)
        
        # Compare both detector (keypoints) and descriptor (semantic) maps
        l_semi = F.mse_loss(semi0, semi1)
        l_desc = F.mse_loss(desc0, desc1)
        
        return l_semi + l_desc


class PerceptualLoss(nn.Module):
    """
    Feature-space distance between enhanced output and a synthetically brightened
    reference using SuperPoint features.
    """
    def __init__(self, backbone: _FrozenBackbone, gamma: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.gamma    = gamma

    def forward(self, enhanced: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = enhanced.shape
        ref = low.pow(self.gamma).clamp(0.0, 1.0)

        # Flatten temporal dim into batch
        enh_2d = enhanced.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        ref_2d = ref.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        semi_enh, desc_enh = self.backbone(enh_2d)
        semi_ref, desc_ref = self.backbone(ref_2d)

        l_semi = F.mse_loss(semi_enh, semi_ref)
        l_desc = F.mse_loss(desc_enh, desc_ref)

        return l_semi + l_desc


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------
class PerceptionLoss(nn.Module):
    """
    Combined SuperPoint-based perception loss.
    """
    def __init__(self,
                 w_temporal:   float = 1.0,  # Increased default weights for SP features
                 w_perceptual: float = 1.0,
                 gamma:        float = 0.4):
        super().__init__()
        backbone          = _FrozenBackbone()
        self.temporal     = SemanticTemporalLoss(backbone)
        self.perceptual   = PerceptualLoss(backbone, gamma=gamma)
        self.w_temporal   = w_temporal
        self.w_perceptual = w_perceptual

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, enhanced: torch.Tensor, low: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        l_temporal   = self.temporal(enhanced)
        l_perceptual = self.perceptual(enhanced, low)
        total        = self.w_temporal * l_temporal + self.w_perceptual * l_perceptual
        return total, {
            "perception_temporal"  : l_temporal.item(),
            "perception_perceptual": l_perceptual.item(),
            "perception_total"     : total.item(),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: ensure weights are downloaded first
    crit   = PerceptionLoss().to(device)

    B, C, D, H, W = 1, 3, 2, 160, 160
    low      = torch.rand(B, C, D, H, W, device=device)
    enhanced = torch.rand(B, C, D, H, W, device=device)

    total, loss_dict = crit(enhanced, low)
    print("SuperPoint PerceptionLoss smoke test:")
    for k, v in loss_dict.items():
        print(f"  {k:<30} {v:.6f}")
    print(f"  finite: {torch.isfinite(total).item()}")
