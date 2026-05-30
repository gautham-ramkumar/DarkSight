import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default loss weights for Zero-3DCE
# ---------------------------------------------------------------------------
# Weights from the Zero-3DCE paper (adapted for our mean-normalized TV implementation)
DEFAULT_WEIGHTS = {
    "color"       : 1.0,
    "exposure"    : 2.5,
    "spatial"     : 4.0,   # Increased for edge sharpening (v2.1)
    "smooth_spa"  : 1.0,
    "smooth_temp" : 1.0,   # Decreased to prevent over-smoothing (v2.1)
    "edge"        : 1.0,
    "ms_ssim"     : 1.0,
}

EXPOSURE_TARGET = 0.6  # Paper uses E=0.6


# ===========================================================================
# Zero-Reference 3D Losses (Self-Supervised)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Color Constancy Loss
# ---------------------------------------------------------------------------
class ColorConstancyLoss(nn.Module):
    """
    Penalises per-frame RGB channel imbalance. Paper Eq. 8:
      L_color = (1/N) Σ_f √(D_RG + D_RB + D_GB + ε)
    where D_xy = (m_x^f − m_y^f)² are per-frame spatial channel means.
    """
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        # Per-frame spatial means: average over H and W only → (B, D)
        mean_r = enhanced[:, 0].mean(dim=[-2, -1])   # (B, D)
        mean_g = enhanced[:, 1].mean(dim=[-2, -1])
        mean_b = enhanced[:, 2].mean(dim=[-2, -1])
        eps = 1e-6
        loss = ((mean_r - mean_g).pow(2) +
                (mean_r - mean_b).pow(2) +
                (mean_g - mean_b).pow(2) + eps).sqrt().mean()
        return loss


# ---------------------------------------------------------------------------
# 2. Exposure Loss
# ---------------------------------------------------------------------------
class ExposureLoss(nn.Module):
    """
    Encourages local patch mean brightness to be close to a target value.
    Computes over 3D tensor but pools SPATIALLY only (kernel_D = 1).
    """
    def __init__(self, patch_size: int = 16, target: float = EXPOSURE_TARGET):
        super().__init__()
        # Pool across H and W, but leave D independent
        self.pool = nn.AvgPool3d((1, patch_size, patch_size))
        self.target = target

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        gray = (0.299 * enhanced[:, 0:1] +
                0.587 * enhanced[:, 1:2] +
                0.114 * enhanced[:, 2:3])
        patch_mean = self.pool(gray)
        return (patch_mean - self.target).abs().mean()


# ---------------------------------------------------------------------------
# 3. Spatial Consistency Loss
# ---------------------------------------------------------------------------
class SpatialConsistencyLoss(nn.Module):
    """
    Preserves spatial differences between adjacent pixels from input to output.
    Extended for 5D tensors (B, 1, D, H, W) using Conv3D.
    """
    def __init__(self):
        super().__init__()
        left  = torch.tensor([[0,  0, 0], [-1, 1, 0], [0,  0, 0]], dtype=torch.float32)
        right = torch.tensor([[0, 0,  0], [0, 1, -1], [0, 0,  0]], dtype=torch.float32)
        up    = torch.tensor([[0, -1, 0], [0,  1, 0], [0,  0, 0]], dtype=torch.float32)
        down  = torch.tensor([[0,  0, 0], [0,  1, 0], [0, -1, 0]], dtype=torch.float32)

        # Stack into (4, 1, 1, 3, 3) kernel for 3D convolution (out_ch=4, in_ch=1, kD=1, kH=3, kW=3)
        kernels = torch.stack([left, right, up, down]).unsqueeze(1).unsqueeze(2)
        self.register_buffer("kernels", kernels)

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        return (0.299 * x[:, 0:1] +
                0.587 * x[:, 1:2] +
                0.114 * x[:, 2:3])

    def forward(self, enhanced: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        enhanced_gray = self._to_gray(enhanced)
        low_gray      = self._to_gray(low)

        d_enhanced = F.conv3d(enhanced_gray, self.kernels, padding=(0, 1, 1))
        d_low      = F.conv3d(low_gray,      self.kernels, padding=(0, 1, 1))

        # Normalise per-sample by the RMS of each directional gradient so the loss
        # measures relative structural change, not absolute gradient magnitude.
        rms_enh = d_enhanced.pow(2).mean(dim=[2, 3, 4], keepdim=True).sqrt().clamp(min=1e-6)
        rms_low = d_low.pow(2).mean(dim=[2, 3, 4], keepdim=True).sqrt().clamp(min=1e-6)
        return ((d_enhanced / rms_enh) - (d_low / rms_low)).pow(2).mean()


# ---------------------------------------------------------------------------
# 4. Illumination Smoothness Loss (Spatial + Temporal TV)
# ---------------------------------------------------------------------------
class IlluminationSmoothnessLoss(nn.Module):
    """
    Encourages the alpha maps to be smooth.
    - Spatial: smooth across H and W (prevents patchy artifacts).
    - Temporal: smooth across D (prevents flickering).
    """
    def forward(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # A: (B, 24, D, H, W)

        # Spatial TV
        grad_x = (A[:, :, :, :, 1:] - A[:, :, :, :, :-1]).abs().mean()
        grad_y = (A[:, :, :, 1:, :] - A[:, :, :, :-1, :]).abs().mean()
        l_spa  = grad_x + grad_y

        # Temporal TV
        if A.shape[2] > 1:
            l_temp = (A[:, :, 1:, :, :] - A[:, :, :-1, :, :]).abs().mean()
        else:
            l_temp = torch.tensor(0.0, device=A.device)

        return l_spa, l_temp


# ---------------------------------------------------------------------------
# 5. Edge Preservation Loss  (Zero-3DCE paper, w4)
# ---------------------------------------------------------------------------
class EdgeLoss(nn.Module):
    """
    Penalises differences between edge maps of enhanced and input.
    Zero-reference: compares output structure to INPUT (no GT needed).
    Forces the model to preserve fine spatial detail while brightening.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # (2, 1, 1, 3, 3) for 3D conv with kD=1
        kernels = torch.stack([sobel_x, sobel_y]).unsqueeze(1).unsqueeze(2)
        self.register_buffer('kernels', kernels)

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, D, H, W) -> gradient magnitude (B, 1, D, H, W)"""
        gray = self._to_gray(x)
        edges = F.conv3d(gray, self.kernels, padding=(0, 1, 1))  # (B, 2, D, H, W)
        return edges.abs().sum(dim=1, keepdim=True)

    def forward(self, enhanced: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        e_low = self._edge_map(low)
        e_enh = self._edge_map(enhanced)
        # Normalise each per-sample by its own mean edge magnitude so the loss
        # measures structural change only, not absolute brightness change.
        norm_low = e_low.mean(dim=[2, 3, 4], keepdim=True).clamp(min=1e-6)
        norm_enh = e_enh.mean(dim=[2, 3, 4], keepdim=True).clamp(min=1e-6)
        # Clamp prevents unstable amplification on textureless inputs where the
        # normalizer sits near the 1e-6 floor and individual pixel values spike.
        e_low_n = (e_low / norm_low).clamp(max=10.0)
        e_enh_n = (e_enh / norm_enh).clamp(max=10.0)
        return (e_enh_n - e_low_n).abs().mean()


# ---------------------------------------------------------------------------
# 6. Multi-Scale SSIM Loss  (Zero-3DCE paper, w6)
# ---------------------------------------------------------------------------
class MSSSIMLoss(nn.Module):
    """
    Multi-scale SSIM between enhanced and original low-light input.
    Zero-reference: measures structural preservation, no GT required.
    Loss = 1 - MS-SSIM  (0 = perfect structural match, higher = worse).
    """
    def __init__(self, scales: int = 3, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.scales = scales
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        window = self.window.expand(x.shape[1], 1, -1, -1)
        mu_x  = F.conv2d(x, window, padding=pad, groups=x.shape[1])
        mu_y  = F.conv2d(y, window, padding=pad, groups=y.shape[1])
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sig_x2 = F.conv2d(x*x, window, padding=pad, groups=x.shape[1]) - mu_x2
        sig_y2 = F.conv2d(y*y, window, padding=pad, groups=y.shape[1]) - mu_y2
        sig_xy = F.conv2d(x*y, window, padding=pad, groups=x.shape[1]) - mu_xy
        num = (2*mu_xy  + C1) * (2*sig_xy  + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (num / den).mean()

    def forward(self, enhanced: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """
        Compare consecutive enhanced frames for temporal consistency.
        Comparing enhanced vs dark input (old approach) was wrong: SSIM includes
        a luminance term, so it penalised the model for brightening the image.
        Comparing frame_0 vs frame_1 in the enhanced clip prevents flickering
        without blocking the brightness increase we want.
        """
        if enhanced.shape[2] < 2:
            return torch.tensor(0.0, device=enhanced.device)

        frame0 = enhanced[:, :, 0, :, :]   # (B, C, H, W)
        frame1 = enhanced[:, :, 1, :, :]   # (B, C, H, W)

        ms_ssim_val = 0.0
        f0, f1 = frame0, frame1
        for _ in range(self.scales):
            ms_ssim_val += self._ssim(f0, f1)
            f0 = F.avg_pool2d(f0, 2)
            f1 = F.avg_pool2d(f1, 2)
        return (1.0 - ms_ssim_val / self.scales).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Combined Zero-3DCE Loss
# ---------------------------------------------------------------------------
class Zero3DCELoss(nn.Module):
    """
    Combined zero-reference loss matching the Zero-3DCE paper.
    All losses are self-supervised (no GT required).
    """
    def __init__(self, weights: dict = None):
        super().__init__()
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

        self.color    = ColorConstancyLoss()
        self.exposure = ExposureLoss()
        self.spatial  = SpatialConsistencyLoss()
        self.smooth   = IlluminationSmoothnessLoss()
        self.edge     = EdgeLoss()
        self.ms_ssim  = MSSSIMLoss()

    def forward(self, A_maps: torch.Tensor, enhanced: torch.Tensor, low: torch.Tensor,
                warp_temporal: torch.Tensor | None = None):
        """
        Args:
            A_maps, enhanced, low: standard Zero-3DCE tensors
            warp_temporal: optional pre-computed flow-compensated temporal loss
                           (WarpTemporalLoss output from raft_warp.py).
                           When provided, replaces the alpha-map temporal TV
                           (smooth_temp) so motion is not penalised.
        """
        l_col         = self.color(enhanced)
        l_exp         = self.exposure(enhanced)
        l_spa         = self.spatial(enhanced, low)
        l_smooth_spa, l_smooth_temp_alpha = self.smooth(A_maps)
        l_edge        = self.edge(enhanced, low)
        l_ms_ssim     = self.ms_ssim(enhanced, low)

        # Use flow-compensated loss when available; fall back to alpha-map TV
        l_smooth_temp = warp_temporal if warp_temporal is not None else l_smooth_temp_alpha

        total = (self.w["color"]       * l_col         +
                 self.w["exposure"]    * l_exp         +
                 self.w["spatial"]     * l_spa         +
                 self.w["smooth_spa"]  * l_smooth_spa  +
                 self.w["smooth_temp"] * l_smooth_temp +
                 self.w["edge"]        * l_edge        +
                 self.w["ms_ssim"]     * l_ms_ssim)

        smooth_temp_val = (l_smooth_temp.item()
                           if isinstance(l_smooth_temp, torch.Tensor)
                           else float(l_smooth_temp))
        loss_dict = {
            "total"      : total.item(),
            "color"      : l_col.item(),
            "exposure"   : l_exp.item(),
            "spatial"    : l_spa.item(),
            "smooth_spa" : l_smooth_spa.item(),
            "smooth_temp": smooth_temp_val,
            "edge"       : l_edge.item(),
            "ms_ssim"    : l_ms_ssim.item(),
        }

        return total, loss_dict


# ===========================================================================
# Validation Metrics (Require Paired GT)
# ===========================================================================

class PSNR(nn.Module):
    def forward(self, enhanced: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(enhanced, gt)
        if mse == 0:
            return torch.tensor(100.0, device=enhanced.device)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size, sigma))

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g      = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.float(), y.float()

        # If 5D, flatten B and D dimensions to treat frames as separate batch items
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x = x.transpose(1, 2).reshape(B*D, C, H, W)
            y = y.transpose(1, 2).reshape(B*D, C, H, W)

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.window_size // 2
        window = self.window.expand(x.shape[1], 1, -1, -1)

        mu_x  = F.conv2d(x, window, padding=pad, groups=x.shape[1])
        mu_y  = F.conv2d(y, window, padding=pad, groups=y.shape[1])
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

        sig_x2  = F.conv2d(x * x, window, padding=pad, groups=x.shape[1]) - mu_x2
        sig_y2  = F.conv2d(y * y, window, padding=pad, groups=y.shape[1]) - mu_y2
        sig_xy  = F.conv2d(x * y, window, padding=pad, groups=x.shape[1]) - mu_xy

        num  = (2 * mu_xy  + C1) * (2 * sig_xy  + C2)
        den  = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (num / den).mean()

    def forward(self, enhanced: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return self._ssim(enhanced, gt)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Zero3DCELoss().to(device)

    # Simulate 5D video batch: Batch=2, Channels=3, D=2 frames, H=256, W=256
    B, C, D, H, W = 2, 3, 2, 256, 256
    low      = torch.rand(B, C, D, H, W).to(device)
    enhanced = torch.rand(B, C, D, H, W).to(device)
    A_maps   = torch.rand(B, 24, D, H, W).to(device)

    total, loss_dict = criterion(A_maps, enhanced, low)

    print("Loss breakdown (Zero-Reference 5D Tensors):")
    print(f"  {'Loss':<20} {'Weight':>8}   {'Value':>10}")
    print(f"  {'-'*45}")
    for name, val in loss_dict.items():
        if name == "total":
            continue
        w = DEFAULT_WEIGHTS.get(name, "-")
        weighted = DEFAULT_WEIGHTS.get(name, 1.0) * val
        print(f"  {name:<20} {w:>8}   {val:>10.4f}   (weighted: {weighted:.4f})")
    print(f"  {'-'*45}")
    print(f"  {'total':<20} {'':>8}   {loss_dict['total']:>10.4f}")
    print()
    print(f"All losses finite: {torch.isfinite(total).item()}")
    print(f"Total loss on device: {device}")
