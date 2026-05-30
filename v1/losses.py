import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroReferenceLoss(nn.Module):
    """
    Standard zero-reference losses used in v1:
    - Color Constancy
    - Exposure Control
    - Spatial Consistency
    - Illumination Smoothness (Spatial + Temporal TV)
    """
    def __init__(self, exposure_target=0.6):
        super().__init__()
        self.exposure_target = exposure_target
        
        # Spatial consistency kernels
        left  = torch.tensor([[0,  0, 0], [-1, 1, 0], [0,  0, 0]], dtype=torch.float32)
        right = torch.tensor([[0, 0,  0], [0, 1, -1], [0, 0,  0]], dtype=torch.float32)
        up    = torch.tensor([[0, -1, 0], [0,  1, 0], [0,  0, 0]], dtype=torch.float32)
        down  = torch.tensor([[0,  0, 0], [0,  1, 0], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("spatial_kernels", torch.stack([left, right, up, down]).unsqueeze(1).unsqueeze(2))

    def _color_loss(self, x):
        mean_r = x[:, 0].mean(dim=[-2, -1])
        mean_g = x[:, 1].mean(dim=[-2, -1])
        mean_b = x[:, 2].mean(dim=[-2, -1])
        return ((mean_r - mean_g).pow(2) + (mean_r - mean_b).pow(2) + (mean_g - mean_b).pow(2) + 1e-6).sqrt().mean()

    def _exposure_loss(self, x):
        avg_intensity = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        # Pool spatially
        pooled = F.avg_pool3d(avg_intensity, (1, 16, 16))
        return (pooled - self.exposure_target).abs().mean()

    def _spatial_loss(self, enh, low):
        enh_gray = (0.299 * enh[:, 0:1] + 0.587 * enh[:, 1:2] + 0.114 * enh[:, 2:3])
        low_gray = (0.299 * low[:, 0:1] + 0.587 * low[:, 1:2] + 0.114 * low[:, 2:3])
        d_enh = F.conv3d(enh_gray, self.spatial_kernels, padding=(0, 1, 1))
        d_low = F.conv3d(low_gray, self.spatial_kernels, padding=(0, 1, 1))
        return (d_enh - d_low).pow(2).mean()

    def _smoothness_loss(self, A):
        # Spatial TV
        grad_x = (A[:, :, :, :, 1:] - A[:, :, :, :, :-1]).abs().mean()
        grad_y = (A[:, :, :, 1:, :] - A[:, :, :, :-1, :]).abs().mean()
        # Temporal TV
        grad_t = (A[:, :, 1:, :, :] - A[:, :, :-1, :, :]).abs().mean() if A.shape[2] > 1 else 0.0
        return grad_x + grad_y, grad_t

    def forward(self, A, enhanced, low):
        l_col = self._color_loss(enhanced)
        l_exp = self._exposure_loss(enhanced)
        l_spa = self._spatial_loss(enhanced, low)
        l_tv_s, l_tv_t = self._smoothness_loss(A)
        
        # v1 Baseline Weights
        total = (1.0 * l_col + 2.0 * l_exp + 2.0 * l_spa + 1.0 * l_tv_s + 2.0 * l_tv_t)
        return total
