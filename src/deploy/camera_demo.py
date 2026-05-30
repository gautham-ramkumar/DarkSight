"""
Real-time low-light enhancement demo — Zero3DCE webcam inference.

Backends:
  pytorch_new   Encoder-decoder Zero3DCE.  Auto-detects illumination head and
                ConvGRU recurrent mode from the loaded checkpoint state-dict.
  pytorch_flat  Legacy flat 7-layer model (pre-Phase-1 best.pth).
  onnx          ONNX Runtime / CUDA (exports/zero3dce.onnx).

What changed from v1:
  Native resolution — no hard 256×256 resize.  H and W are padded to the
  nearest multiple of 8 (3× spatial downsampling constraint) and stripped
  before display, so the full camera resolution is enhanced.

  Proper D=2 temporal input — a frame buffer stacks (prev, cur) for batch
  mode models; recurrent mode models receive D=1 per frame and carry
  ConvGRU hidden state across the sequence.

  Illumination panel (--illum) — when the checkpoint includes the
  illumination head, a third panel shows the per-pixel illumination map
  as a false-colour (INFERNO) heatmap.  The brightness bar updates from
  the model's illumination estimate when available.

  Camera-disconnect guard — exits cleanly after ~1.5 s of missed frames
  instead of spinning forever.

Keyboard:
  q / ESC   Quit
  + / -     Raise / lower brightness threshold
  r         Reset temporal state (ConvGRU hidden + frame buffer)
  s         Save composite screenshot to test_outputs/demo/
  f         Toggle fullscreen
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.model import Zero3DCE
from core.utils import pad_to_align, unpad


# ---------------------------------------------------------------------------
# Legacy flat model — matches old best.pth from pre-Phase-1 runs
# ---------------------------------------------------------------------------
class _FlatZero3DCE(nn.Module):
    def __init__(self, n_iter: int = 8):
        super().__init__()
        self.n_iter = n_iter
        self.conv1 = nn.Sequential(nn.Conv3d(3,  32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Conv3d(64, 3 * n_iter, 3, padding=1)

    def forward(self, x):
        f1 = self.conv1(x); f2 = self.conv2(f1)
        f3 = self.conv3(f2); f4 = self.conv4(f3)
        f5 = self.conv5(torch.cat([f3, f4], dim=1))
        f6 = self.conv6(torch.cat([f2, f5], dim=1))
        A  = torch.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        out = x
        for i in range(self.n_iter):
            out = out + A[:, i*3:(i+1)*3] * (out - out**2)
        return A, out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Backend wrappers — uniform interface: run / is_recurrent / illum_map / reset_state
# ---------------------------------------------------------------------------
class PyTorchBackend:
    """
    Wraps _FlatZero3DCE (pytorch_flat) or Zero3DCE (pytorch_new).
    For Zero3DCE checkpoints, auto-configures illumination head and recurrent
    mode by scanning state-dict keys — no flags needed.
    Carries ConvGRU hidden state across frames in recurrent mode.
    """
    def __init__(self, use_flat: bool, checkpoint: str | None, device: torch.device):
        self.device      = device
        self._hidden     = None
        self._last_illum = None

        if use_flat:
            self.model  = _FlatZero3DCE().to(device)
            arch_label  = "flat"
        else:
            has_illum = has_gru = False
            if checkpoint:
                probe     = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state     = probe.get("model", probe)
                has_illum = any(k.startswith("illum_head") for k in state)
                has_gru   = any(k.startswith("gru")        for k in state)
            self.model = Zero3DCE(
                predict_illumination = has_illum,
                use_recurrent        = has_gru,
            ).to(device)
            arch_label = (("recurrent" if has_gru else "batch") +
                          ("+illum"    if has_illum else ""))

        if checkpoint:
            ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state)
            epoch = ckpt.get("epoch",    "?")
            psnr  = ckpt.get("val_psnr", float("nan"))
            self._label = f"PyTorch ({arch_label} | ep={epoch} | {psnr:.2f}dB)"
        else:
            self._label = f"PyTorch ({arch_label} | random weights)"

        self.model.eval()
        print(f"[backend] {self._label}")

    @property
    def is_recurrent(self) -> bool:
        return getattr(getattr(self, "model", None), "use_recurrent", False)

    @property
    def illum_map(self) -> torch.Tensor | None:
        return self._last_illum

    def reset_state(self):
        self._hidden = None

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1,3,D,H',W')  →  (1,3,1,H',W') enhanced current frame."""
        with torch.no_grad():
            out = self.model(x, self._hidden) if self.is_recurrent else self.model(x)

        if len(out) == 4:
            _, enhanced, self._last_illum, self._hidden = out
        elif len(out) == 3:
            _, enhanced, self._last_illum = out
            self._hidden = None
        else:
            _, enhanced = out
            self._last_illum = self._hidden = None

        return enhanced[:, :, -1:]   # always return (1, 3, 1, H', W')

    @property
    def label(self) -> str:
        return self._label


class ONNXBackend:
    def __init__(self, onnx_path: str, device: torch.device):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda" and
               "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.sess   = ort.InferenceSession(onnx_path, providers=providers)
        self.device = device
        
        # Detect if model is recurrent by checking inputs
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self._is_recurrent = "hidden_in" in self.input_names
        self.hidden_state = None
        self._latest_illum = None
        
        prov        = providers[0].replace("ExecutionProvider", "")
        self._label = f"ONNX-RT/{prov}"
        print(f"[backend] {self._label}  ({onnx_path})")
        if self._is_recurrent:
            print("  [mode] recurrent (ConvGRU)")
        else:
            print("  [mode] batch (D=2)")
        print("[backend] NOTE: re-export with export_trt.py for dynamic-resolution support.")

    @property
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @property
    def illum_map(self) -> torch.Tensor | None:
        return self._latest_illum

    def reset_state(self):
        self.hidden_state = None
        self._latest_illum = None

    def run(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np
        x_np = x.cpu().numpy()
        
        if self._is_recurrent:
            if self.hidden_state is None:
                # Shape: (B, 32, H//8, W//8)
                B, C, D, H, W = x.shape
                self.hidden_state = np.zeros((B, 32, H//8, W//8), dtype=np.float32)
            
            feeds = {"input": x_np, "hidden_in": self.hidden_state}
            outputs = self.sess.run(None, feeds)
            # outputs: [alpha, enhanced, illum, hidden_out]
            enhanced = torch.from_numpy(outputs[1]).to(self.device)
            self._latest_illum = torch.from_numpy(outputs[2]).to(self.device)
            self.hidden_state = outputs[3]
        else:
            outputs = self.sess.run(None, {"input": x_np})
            # outputs: [alpha, enhanced, illum]
            enhanced = torch.from_numpy(outputs[1]).to(self.device)
            self._latest_illum = torch.from_numpy(outputs[2]).to(self.device)
            
        return enhanced[:, :, -1:]

    @property
    def label(self) -> str:
        return self._label


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing — Speed & Quality Optimisations (v2.2)
# ---------------------------------------------------------------------------
def preprocess(bgr: np.ndarray, device: torch.device, downsample: bool = True
               ) -> tuple[torch.Tensor, int, int, np.ndarray | None]:
    """
    Optimised preprocessing:
      1. Converts to YCbCr to decouple Luminance from Color (fixes tint).
      2. Optional downsample (e.g. 720p -> 360p) for 4x FPS boost.
      3. Returns Y-tensor, padding info, and original CbCr for reconstruction.
    """
    h, w = bgr.shape[:2]
    
    # Decouple Luma/Chroma using YCrCb (OpenCV's version of YCbCr)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # We only enhance the Y channel
    if downsample:
        y_proc = cv2.resize(y, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    else:
        y_proc = y

    # Convert to 1-channel RGB-like tensor for Zero3DCE (it expects 3 channels)
    # We duplicate Y across 3 channels so the model's filters work as trained.
    y_3ch = cv2.merge([y_proc, y_proc, y_proc])
    
    t = torch.from_numpy(y_3ch).float().div(255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)      # (1, 3, 1, H/2, W/2)
    t, pad_h, pad_w = pad_to_align(t, align=8)
    
    return t.to(device, non_blocking=True), pad_h, pad_w, ycrcb


def postprocess(tensor: torch.Tensor, display_h: int, display_w: int,
                pad_h: int, pad_w: int, 
                orig_ycrcb: np.ndarray, denoise: bool = True) -> np.ndarray:
    """
    Optimised postprocessing:
      1. Extracts enhanced Y.
      2. Recombines with original Cr/Cb (perfect color preservation).
      3. Optional bilateral filter (denoising).
      4. Upscales to display resolution.
    """
    # 1. Extract and unpad the enhanced Y (take only 1 channel of the 3)
    frame = tensor[0, 0:1, 0].clamp(0, 1)        # (1, H', W')
    frame = unpad(frame, pad_h, pad_w)           # (1, H_proc, W_proc)
    y_enh = (frame[0].cpu().numpy() * 255).astype(np.uint8)
    
    # 2. Recombine with original chroma
    h_orig, w_orig = orig_ycrcb.shape[:2]
    if y_enh.shape != (h_orig, w_orig):
        y_enh = cv2.resize(y_enh, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    _, cr, cb = cv2.split(orig_ycrcb)
    out_ycrcb = cv2.merge([y_enh, cr, cb])
    out_bgr   = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    # 3. Optional denoising (Bilateral filter is great for edge-preserving smoothing)
    if denoise:
        out_bgr = cv2.bilateralFilter(out_bgr, d=5, sigmaColor=35, sigmaSpace=35)

    # 4. Final display resize
    if (out_bgr.shape[1], out_bgr.shape[0]) != (display_w, display_h):
        out_bgr = cv2.resize(out_bgr, (display_w, display_h))
        
    return out_bgr


def illum_to_panel(illum: torch.Tensor, display_h: int, display_w: int,
                   pad_h: int = 0, pad_w: int = 0) -> np.ndarray:
    """(1,1,1,H',W') illumination map  →  INFERNO false-colour BGR panel."""
    lmap = illum[0, 0, 0].clamp(0, 1)
    lmap = unpad(lmap, pad_h, pad_w)
    lmap = (lmap.cpu().numpy() * 255).astype(np.uint8)
    lmap = cv2.applyColorMap(lmap, cv2.COLORMAP_INFERNO)
    if (lmap.shape[1], lmap.shape[0]) != (display_w, display_h):
        lmap = cv2.resize(lmap, (display_w, display_h))
    return lmap


def mean_brightness(bgr: np.ndarray) -> float:
    """CIE L* mean brightness ∈ [0, 1] — better than raw pixel mean."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean()) / 255.0


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def put_text(img: np.ndarray, text: str, pos, color=(255, 255, 255),
             scale: float = 0.55, thickness: int = 1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color,    thickness,     cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------
_MAX_FAIL = 30   # consecutive cap.read() failures → ~1.5 s before abort

def run_demo(backend, cam_id: int, display_w: int, display_h: int,
             threshold: float, device: torch.device, out_dir: Path,
             show_illum: bool = False):

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{cam_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Aspect ratio preservation ────────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read from camera /dev/video{cam_id}")

    cam_h, cam_w = first_frame.shape[:2]
    aspect = cam_w / cam_h
    if display_h == 480 and display_w == 640: # defaults, or not specified
        display_h = int(display_w / aspect)

    n_panels  = 3 if show_illum else 2
    win       = "Zero3DCE  [q=quit  +/-=threshold  r=reset  s=save  f=fullscreen]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w * n_panels + 4 * (n_panels - 1), display_h)

    full        = False
    fps_smooth  = 30.0
    ema_a       = 0.1
    prev_tensor = None          # frame buffer for batch-mode D=2 input
    pad_h = pad_w = 0
    _fail_count   = 0
    prev_brightness = None

    print(f"\nBackend    : {backend.label}")
    print(f"Camera     : /dev/video{cam_id}  ({cam_w}×{cam_h} native)")
    print(f"Display    : {display_w}×{display_h} per panel")
    print(f"Mode       : {'recurrent (ConvGRU)' if backend.is_recurrent else 'batch (D=2 pairs)'}")
    print(f"Threshold  : {threshold:.2f}  (L* < threshold → enhance)")
    print(f"Illum panel: {show_illum}")
    print("Controls   : +/- threshold | r reset state | s save | f fullscreen | q quit\n")

    while True:
        ret, frame = cap.read()

        # ── Disconnect guard ─────────────────────────────────────────────────
        if not ret:
            _fail_count += 1
            if _fail_count >= _MAX_FAIL:
                print("\n[warn] Camera stream lost — exiting.")
                backend.reset_state()
                break
            time.sleep(0.05)
            continue
        _fail_count = 0

        # ── Preprocess (v2.2: Y-only + 2x downsample) ────────────────────────
        t_cur, pad_h, pad_w, orig_ycrcb = preprocess(frame, device, downsample=True)

        if backend.is_recurrent:
            x = t_cur                                   # (1, 3, 1, H/2, W/2)
        else:
            if prev_tensor is None:
                prev_tensor = t_cur
            x = torch.cat([prev_tensor, t_cur], dim=2)

        # ── Decision based on raw frame (Reliable CIE L*) ────────────────────
        brightness = mean_brightness(frame)
        is_dark    = brightness < threshold
        src_label  = "CIE L*"

        # ── Inference ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if is_dark:
            enhanced_t = backend.run(x)
        else:
            if show_illum:
                _ = backend.run(x)
            enhanced_t = t_cur

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        fps_smooth = (1 - ema_a) * fps_smooth + ema_a * (1000 / max(elapsed_ms, 1))

        # ── Scene cut detection ──────────────────────────────────────────────
        if prev_brightness is not None:
            if abs(brightness - prev_brightness) > 0.3:
                backend.reset_state()
                print(f"[info] Scene cut detected (Δ={abs(brightness - prev_brightness):.2f}) — state reset.")
        prev_brightness = brightness

        # ── Postprocess (v2.2: Recombine with orig CbCr + Denoise) ───────────
        illum = backend.illum_map
        if is_dark:
            show_right = postprocess(enhanced_t, display_h, display_w, pad_h, pad_w, 
                                     orig_ycrcb, denoise=True)
        else:
            show_right = cv2.resize(frame, (display_w, display_h))

        prev_tensor = t_cur   # always advance frame buffer

        # ── Panels ───────────────────────────────────────────────────────────
        show_left = cv2.resize(frame, (display_w, display_h))

        put_text(show_left,
                 f"Input  L*={brightness:.2f}  {'[dark]' if is_dark else '[bright]'}",
                 (8, 24), (80, 200, 255))
        bar_w = int(display_w * min(brightness, 1.0))
        cv2.rectangle(show_left, (0, display_h - 8), (bar_w, display_h),
                      (0, 80, 255) if is_dark else (0, 220, 80), -1)
        put_text(show_left, f"thr={threshold:.2f}",
                 (8, display_h - 14), (200, 200, 200), scale=0.42)

        if is_dark:
            put_text(show_right, f"Enhanced ({src_label})", (8, 24), (80, 255, 80))
        else:
            put_text(show_right, f"Passthrough ({src_label})", (8, 24), (80, 220, 80))

        put_text(show_right, f"{fps_smooth:.0f} FPS  {elapsed_ms:.1f} ms",
                 (8, display_h - 12), (255, 255, 80))
        put_text(show_right, backend.label, (8, display_h - 30),
                 (200, 200, 200), scale=0.38)

        div    = np.zeros((display_h, 4, 3), dtype=np.uint8)
        panels = [show_left, div, show_right]

        if show_illum:
            if illum is not None:
                ip = illum_to_panel(illum, display_h, display_w, pad_h, pad_w)
                put_text(ip, "Illumination L  (model)", (8, 24), (255, 200, 50))
            else:
                ip = np.full((display_h, display_w, 3), 40, dtype=np.uint8)
                put_text(ip, "No illumination head", (8, display_h // 2), (180, 180, 180))
            panels += [div, ip]

        cv2.imshow(win, np.hstack(panels))

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(threshold + 0.02, 1.0)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(threshold - 0.02, 0.0)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('r'):
            backend.reset_state()
            prev_tensor = None
            print("Temporal state reset.")
        elif key == ord('s'):
            p = out_dir / f"capture_{int(time.time())}.png"
            cv2.imwrite(str(p), np.hstack(panels))
            print(f"Saved: {p}")
        elif key == ord('f'):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    src_dir  = Path(__file__).resolve().parent.parent
    proj_dir = src_dir.parent

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Real-time Zero3DCE webcam demo",
    )
    parser.add_argument("--backend",   choices=["pytorch_new", "pytorch_flat", "onnx"],
                        default="pytorch_flat")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (defaults to checkpoints/best.pth)")
    parser.add_argument("--onnx",       type=str,
                        default=str(proj_dir / "exports" / "zero3dce.onnx"))
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--width",      type=int,   default=640,
                        help="Display width per panel")
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--threshold",  type=float, default=0.35,
                        help="CIE L* threshold: enhance if mean brightness < this")
    parser.add_argument("--illum",      action="store_true",
                        help="Show illumination map as a 3rd panel (pytorch_new only)")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = proj_dir / "test_outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    ckpt_default = str(proj_dir / "checkpoints" / "best.pth")

    if args.backend == "pytorch_flat":
        ckpt = args.checkpoint or ckpt_default
        if not Path(ckpt).exists():
            print(f"[warn] {ckpt} not found — using random weights.")
            ckpt = None
        backend = PyTorchBackend(use_flat=True, checkpoint=ckpt, device=device)

    elif args.backend == "pytorch_new":
        ckpt = args.checkpoint or ckpt_default
        if not Path(ckpt).exists():
            print(f"[warn] {ckpt} not found — using random weights.")
            ckpt = None
        backend = PyTorchBackend(use_flat=False, checkpoint=ckpt, device=device)

    elif args.backend == "onnx":
        if not Path(args.onnx).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {args.onnx}\n"
                "Run:  python3 export_trt.py --checkpoint ../checkpoints/best.pth"
            )
        backend = ONNXBackend(args.onnx, device)

    run_demo(backend,
             cam_id     = args.camera,
             display_w  = args.width,
             display_h  = args.height,
             threshold  = args.threshold,
             device     = device,
             out_dir    = out_dir,
             show_illum = args.illum)


if __name__ == "__main__":
    main()
