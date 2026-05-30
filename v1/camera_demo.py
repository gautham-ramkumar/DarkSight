"""
DarkSight v1: Legacy Baseline Demo (Webcam Inference)
Uses the original 7-layer flat 3D convolutional network.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure we can import from the root and v1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v1.model import FlatZero3DCE
from src.core.utils import pad_to_align, unpad

def mean_brightness(bgr: np.ndarray) -> float:
    """CIE L* mean brightness ∈ [0, 1]."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean()) / 255.0

def put_text(img: np.ndarray, text: str, pos, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def run_v1_demo(cam_id, checkpoint_path, threshold, display_w, display_h):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load v1 model
    model = FlatZero3DCE().to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading v1 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
    else:
        print("Using random weights (v1 baseline).")
    model.eval()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {cam_id}")
        return

    prev_tensor = None
    win_name = "DarkSight v1 (Baseline Flat Architecture)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("Controls: q quit | +/- threshold")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocess
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_cur = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        t_cur, pad_h, pad_w = pad_to_align(t_cur, align=8)
        t_cur = t_cur.to(device)

        # CIE L* for decision
        brightness = mean_brightness(frame)
        is_dark = brightness < threshold

        t0 = time.perf_counter()
        if is_dark:
            # v1 expects D=2 input pairs
            if prev_tensor is None: prev_tensor = t_cur
            x = torch.cat([prev_tensor, t_cur], dim=2)
            
            with torch.no_grad():
                _, enhanced = model(x)
            
            # Extract last frame
            out_frame = enhanced[0, :, -1].permute(1, 2, 0).cpu().numpy()
            out_frame = unpad(torch.from_numpy(out_frame).permute(2, 0, 1), pad_h, pad_w).permute(1, 2, 0).numpy()
            out_bgr = cv2.cvtColor((out_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            out_bgr = frame
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        prev_tensor = t_cur

        # Visualization
        show_left = cv2.resize(frame, (display_w, display_h))
        show_right = cv2.resize(out_bgr, (display_w, display_h))
        
        put_text(show_left, f"Input (L*={brightness:.2f})", (10, 30))
        put_text(show_right, f"v1 Enhanced ({elapsed_ms:.1f}ms)", (10, 30), color=(80, 255, 80))
        
        cv2.imshow(win_name, np.hstack([show_left, show_right]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in (ord('+'), ord('=')): threshold += 0.05
        elif key == ord('-'): threshold -= 0.05

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    
    run_v1_demo(args.camera, args.checkpoint, args.threshold, 640, 480)
