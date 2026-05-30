import torch
import torch.nn as nn
import torch.nn.functional as F


class DS3DConv(nn.Module):
    """Depthwise-separable 3D conv: spatial depthwise + 1×1×1 pointwise."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SpatialAttention3D(nn.Module):
    """
    Channel-wise avg+max pool → concat → 7×7×7 conv → sigmoid.
    Guides the network to focus on under-exposed spatial regions.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class EncoderBlock(nn.Module):
    """DS3DConv → ReLU → SpatialAttention3D → MaxPool3d (spatial only)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(DS3DConv(in_ch, out_ch), nn.ReLU(inplace=True))
        self.attn = SpatialAttention3D()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor):
        x = self.attn(self.conv(x))
        return self.pool(x), x   # (pooled, skip)


class ConvGRUCell(nn.Module):
    """
    2D spatial ConvGRU cell for the recurrent bottleneck.

    Processes one frame's spatial feature map (H/8 × W/8) at a time and
    maintains persistent hidden state across the sequence.  Temporal memory
    lives entirely here — the encoder/decoder 3D convolutions operate on
    D=1 slices (effectively 2D) in recurrent mode.

    Standard GRU gating (Cho et al. 2014) with 2D spatial convolutions:
        r  = σ(W_r ∗ [x, h])
        z  = σ(W_z ∗ [x, h])
        n  = tanh(W_n ∗ [x, r ⊙ h])
        h' = (1 − z) ⊙ h + z ⊙ n
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad  = kernel_size // 2
        both = in_channels + hidden_channels
        self.reset_gate  = nn.Conv2d(both, hidden_channels, kernel_size, padding=pad)
        self.update_gate = nn.Conv2d(both, hidden_channels, kernel_size, padding=pad)
        self.candidate   = nn.Conv2d(both, hidden_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch,     Hs, Ws)  current bottleneck features
        h : (B, hidden_ch, Hs, Ws)  previous hidden state
        → new hidden state (B, hidden_ch, Hs, Ws)
        """
        xh = torch.cat([x, h], dim=1)
        r  = torch.sigmoid(self.reset_gate(xh))
        z  = torch.sigmoid(self.update_gate(xh))
        n  = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


class Zero3DCE(nn.Module):
    """
    DarkSight v2: encoder-decoder with optional ConvGRU recurrent bottleneck
    and optional illumination prediction head.

    ── Forward modes ──────────────────────────────────────────────────────────

    Batch mode  (use_recurrent=False, default)
        All D frames processed together through 3D convolutions.
        Temporal info flows via 3D conv kernels.  Fast, D=2 pairs.

    Recurrent mode  (use_recurrent=True)
        Frames processed one at a time (D=1 through 3D convs = effectively 2D).
        ConvGRU at the bottleneck accumulates temporal memory across the sequence.
        Supports arbitrarily long sequences: tunnel entry/exit, sustained
        lighting transitions, etc.
        Truncated BPTT: hidden state detached every `tbptt_steps` frames to
        keep gradient graphs tractable.

    ── Optional heads ─────────────────────────────────────────────────────────

    Illumination head  (predict_illumination=True)
        Small DS3DConv branch off the bottleneck.  Predicts per-pixel
        illumination L ∈ (0, 1] at full resolution (Retinex: I = R × L).
        Auxiliary output — does not modify the enhancement path.
        Used by camera_demo to make spatially-aware bright/dark decisions.

    ── Interface ──────────────────────────────────────────────────────────────

    Input  : (B, 3, D, H, W)  — H and W must be divisible by 8
    Output : (A, enhanced, illum_map, hidden_state)
        A            (B, 3·n_iter, D, H, W)
        enhanced     (B, 3,        D, H, W)
        illum_map    (B, 1,        D, H, W) or None
        hidden_state (B, 32, H/8, W/8)     or None  (None in batch mode)
    """

    def __init__(self,
                 n_iter:               int  = 8,
                 predict_illumination: bool = False,
                 use_recurrent:        bool = False,
                 tbptt_steps:          int  = 4):
        super().__init__()
        self.n_iter               = n_iter
        self.predict_illumination = predict_illumination
        self.use_recurrent        = use_recurrent
        self.tbptt_steps          = tbptt_steps

        self.enc1 = EncoderBlock(3,  32)   # skip at H,   pool at H/2
        self.enc2 = EncoderBlock(32, 32)   # skip at H/2, pool at H/4
        self.enc3 = EncoderBlock(32, 32)   # skip at H/4, pool at H/8

        self.bottleneck = nn.Sequential(DS3DConv(32, 32), nn.ReLU(inplace=True))

        self.dec1 = nn.Sequential(DS3DConv(64, 32), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(DS3DConv(64, 32), nn.ReLU(inplace=True))
        self.dec3 = DS3DConv(64, 3 * n_iter)

        nn.init.constant_(self.dec3.pw.bias, 0.0)

        if predict_illumination:
            self.illum_head = nn.Sequential(
                DS3DConv(32, 16), nn.ReLU(inplace=True),
                DS3DConv(16,  1),
                nn.Sigmoid(),
            )

        if use_recurrent:
            self.gru = ConvGRUCell(32, 32)

    # ── Shared sub-routines ───────────────────────────────────────────────────

    def _decode(self, b, s1, s2, s3) -> torch.Tensor:
        """Decoder: bottleneck → alpha maps (B, 3·n_iter, D, H, W)."""
        d1 = F.interpolate(b,  scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, s3], dim=1))
        d2 = F.interpolate(d1, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d3 = F.interpolate(d2, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        return torch.tanh(self.dec3(torch.cat([d3, s1], dim=1)))

    def _enhance(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Apply LEQ curve: x = x + A_i * (x - x²) for each of n_iter steps."""
        for i in range(self.n_iter):
            x = x + A[:, i*3:(i+1)*3] * (x - x**2)
        return x.clamp(0.0, 1.0)

    def _illum(self, b: torch.Tensor, target_size) -> torch.Tensor | None:
        """Illumination map from bottleneck features; None when head is disabled."""
        if not self.predict_illumination:
            return None
        up = F.interpolate(b, size=target_size, mode='trilinear', align_corners=False)
        return self.illum_head(up)

    # ── Forward paths ─────────────────────────────────────────────────────────

    def _forward_batch(self, x: torch.Tensor):
        """All D frames processed simultaneously through 3D convolutions."""
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        b      = self.bottleneck(p3)

        A        = self._decode(b, s1, s2, s3)
        enhanced = self._enhance(x, A)
        illum    = self._illum(b, x.shape[2:])
        return A, enhanced, illum, None

    def _forward_recurrent(self, x: torch.Tensor, hidden: torch.Tensor | None):
        """Sequential frame-by-frame processing with ConvGRU at the bottleneck."""
        B, C, D, H, W = x.shape

        if hidden is None:
            # Bottleneck spatial size after 3× MaxPool(1,2,2)
            hidden = x.new_zeros(B, 32, H // 8, W // 8)

        all_A, all_enhanced = [], []
        illum_out = None

        for d in range(D):
            # Truncated BPTT: detach hidden state every tbptt_steps frames
            if d > 0 and d % self.tbptt_steps == 0:
                hidden = hidden.detach()

            frame   = x[:, :, d:d+1]          # (B, C, 1, H, W)
            p1, s1  = self.enc1(frame)
            p2, s2  = self.enc2(p1)
            p3, s3  = self.enc3(p2)
            b       = self.bottleneck(p3)      # (B, 32, 1, H/8, W/8)

            # GRU update: squeeze temporal dim → apply cell → unsqueeze back
            hidden  = self.gru(b[:, :, 0], hidden)   # (B, 32, H/8, W/8)
            b       = hidden.unsqueeze(2)             # (B, 32, 1, H/8, W/8)

            A         = self._decode(b, s1, s2, s3)
            enhanced  = self._enhance(frame, A)
            illum_out = self._illum(b, frame.shape[2:])

            all_A.append(A)
            all_enhanced.append(enhanced)

        return (torch.cat(all_A,        dim=2),
                torch.cat(all_enhanced, dim=2),
                illum_out,
                hidden)

    # ── Public forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            x      : (B, 3, D, H, W) input clip — H,W divisible by 8
            hidden : (B, 32, H/8, W/8) previous GRU hidden state, or None.
                     Ignored in batch mode (use_recurrent=False).

        Returns:
            A, enhanced, illum_map, hidden_state
            hidden_state is None in batch mode.
        """
        if self.use_recurrent:
            return self._forward_recurrent(x, hidden)
        return self._forward_batch(x)


# ---------------------------------------------------------------------------
# Smoke test + parameter count + FPS benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("── Batch mode (default) ──────────────────────────────────────────")
    model_batch = Zero3DCE().to(device)
    total = sum(p.numel() for p in model_batch.parameters())
    print(f"Parameters (batch)    : {total:,}")

    x = torch.rand(1, 3, 2, 256, 256, device=device)
    with torch.no_grad():
        A, enh, illum, h = model_batch(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"A      : {tuple(A.shape)}")
    print(f"enh    : {tuple(enh.shape)}  range [{enh.min():.3f}, {enh.max():.3f}]")
    print(f"illum  : {illum}  (None — head disabled)")
    print(f"hidden : {h}  (None — batch mode)")

    print()
    print("── Recurrent mode (D=8, ConvGRU) ────────────────────────────────")
    model_rec = Zero3DCE(use_recurrent=True, predict_illumination=True).to(device)
    total_rec = sum(p.numel() for p in model_rec.parameters())
    print(f"Parameters (recurrent): {total_rec:,}")

    x8 = torch.rand(1, 3, 8, 256, 256, device=device)
    with torch.no_grad():
        A8, enh8, illum8, h8 = model_rec(x8)
    print(f"Input  : {tuple(x8.shape)}")
    print(f"A      : {tuple(A8.shape)}")
    print(f"enh    : {tuple(enh8.shape)}")
    print(f"illum  : {tuple(illum8.shape)}")
    print(f"hidden : {tuple(h8.shape)}  (GRU state, H/8 × W/8)")

    print()
    print("── FPS benchmark (batch mode, 256×256, D=2) ─────────────────────")
    model_bench = Zero3DCE().to(device)
    x_b = torch.rand(1, 3, 2, 256, 256, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model_bench(x_b)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model_bench(x_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fps = (n_runs * 2) / (time.perf_counter() - t0)
    print(f"  256×256 D=2  →  {fps:.1f} FPS")
    print(f"  Real-time (≥30 FPS) : {'YES' if fps >= 30 else 'NO'}")
