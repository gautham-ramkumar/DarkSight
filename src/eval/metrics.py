"""
Phase 4 perception metrics for Zero-3DCE validation.

Single-frame metrics (LOL val set, paired low/high):
    YChannelPSNR   — luminance PSNR (BT.601), typically 1-2 dB above RGB PSNR
    LPIPSMetric    — perceptual distance (AlexNet), lower is better
    DetectionMAP   — YOLOv8n pseudo-GT mAP@IoU=0.5, measures detection fidelity

Video / temporal metrics (DarkVideo consecutive pairs):
    ORBMetrics     — keypoint count + homography inlier ratio (SLAM proxy)
    FlowConsistency — RAFT EPE vs gamma-brightened reference (temporal drift)

All classes are lazy-loading: optional dependencies (lpips, ultralytics, cv2,
torchvision optical_flow) are imported only inside __init__, so this module can
always be imported even if a package is absent.  Missing-package errors are
raised at construction time with an install hint, not at import time.

Interface:
    Simple scalar metrics (YChannelPSNR, LPIPSMetric):
        value = metric(enhanced, gt)   → float tensor, per batch

    Accumulating metrics (DetectionMAP, ORBMetrics, FlowConsistency):
        metric.reset()
        for batch in loader:
            metric.update(...)
        result_dict = metric.compute()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) [0,1] → (B, 1, H, W) luminance (BT.601)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes: (N, 4) [x1, y1, x2, y2]
    Returns: (N, M)
    """
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-6)


def _compute_ap50(pred: torch.Tensor, gt_boxes: torch.Tensor,
                  iou_threshold: float = 0.5) -> float:
    """
    Class-agnostic AP@IoU=0.5 for a single image.

    pred:     (N, 5) [x1, y1, x2, y2, conf]  — sorted by conf descending
    gt_boxes: (M, 4) [x1, y1, x2, y2]
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    if len(pred) == 0:
        return 0.0

    order    = pred[:, 4].argsort(descending=True)
    pred     = pred[order]
    matched  = set()
    tp_list, fp_list = [], []

    ious = _box_iou(pred[:, :4], gt_boxes)  # (N, M)
    for i in range(len(pred)):
        best_iou, best_j = ious[i].max(0)
        j = best_j.item()
        if best_iou.item() >= iou_threshold and j not in matched:
            tp_list.append(1); fp_list.append(0)
            matched.add(j)
        else:
            tp_list.append(0); fp_list.append(1)

    tp  = torch.tensor(tp_list, dtype=torch.float32).cumsum(0)
    fp  = torch.tensor(fp_list, dtype=torch.float32).cumsum(0)
    rec = tp / (len(gt_boxes) + 1e-6)
    pre = tp / (tp + fp + 1e-6)

    # Prepend (0, 1) sentinel for trapezoidal integration
    rec = torch.cat([torch.zeros(1), rec])
    pre = torch.cat([torch.ones(1),  pre])
    return torch.trapz(pre, rec).item()


# ---------------------------------------------------------------------------
# 1. Y-channel PSNR
# ---------------------------------------------------------------------------
class YChannelPSNR(nn.Module):
    """
    PSNR on the luminance (Y) channel only — matches common paper conventions.
    Typically 1–2 dB higher than full-RGB PSNR.

    Requires: nothing beyond PyTorch
    """

    def forward(self, enhanced: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced, gt: (B, 3, H, W) float in [0, 1]
        Returns:
            scalar PSNR in dB
        """
        y_enh = _rgb_to_y(enhanced)
        y_gt  = _rgb_to_y(gt)
        mse   = F.mse_loss(y_enh, y_gt)
        if mse == 0:
            return torch.tensor(100.0, device=enhanced.device)
        return 20.0 * torch.log10(1.0 / mse.sqrt())


# ---------------------------------------------------------------------------
# 2. LPIPS (AlexNet)
# ---------------------------------------------------------------------------
class LPIPSMetric(nn.Module):
    """
    Learned Perceptual Image Patch Similarity using AlexNet.
    Lower is better.  Frozen weights, always eval.

    Requires: pip install lpips
    """

    def __init__(self):
        super().__init__()
        try:
            import lpips as _lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPSMetric requires the lpips package.\n"
                "  pip install lpips\n"
                f"  Original error: {exc}"
            ) from exc

        self.fn = _lpips.LPIPS(net='alex', verbose=False)
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, enhanced: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced, gt: (B, 3, H, W) float in [0, 1]
        Returns:
            mean LPIPS distance (scalar)
        """
        # lpips expects [-1, 1]; normalize=True converts from [0,1]
        return self.fn(enhanced, gt, normalize=True).mean()


# ---------------------------------------------------------------------------
# 3. Detection mAP@0.5 (YOLOv8 pseudo-GT)
# ---------------------------------------------------------------------------
class DetectionMAP:
    """
    Class-agnostic mAP@IoU=0.5 using YOLOv8n as both detector and pseudo-GT.

    Workflow per image pair (enhanced, gt):
      1. Run YOLOv8n on the GT (bright) frame → pseudo-ground-truth boxes
      2. Run YOLOv8n on the enhanced frame    → predictions
      3. Match predictions to pseudo-GT at IoU ≥ 0.5 → AP per image
    Averages AP across all images.

    Measures: does enhancement preserve the detections visible in bright frames?

    Requires: pip install ultralytics
    """

    def __init__(self, model_name: str = 'yolov8n.pt', device: torch.device = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "DetectionMAP requires the ultralytics package.\n"
                "  pip install ultralytics\n"
                f"  Original error: {exc}"
            ) from exc

        self.yolo   = YOLO(model_name)
        self.device = device or torch.device('cpu')
        self._aps   = []

    def reset(self):
        self._aps = []

    def _run_yolo(self, frames: torch.Tensor) -> list[torch.Tensor]:
        """
        Run YOLOv8 on a batch of frames.

        Args:
            frames: (B, 3, H, W) float in [0, 1]
        Returns:
            list of B tensors, each (N, 4) [x1,y1,x2,y2] — raw box coords
        """
        imgs_np = (frames.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        imgs_np = imgs_np.transpose(0, 2, 3, 1)    # (B, H, W, 3) RGB uint8
        results = self.yolo(list(imgs_np), verbose=False)
        out = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu()          # (N, 4)
                confs = r.boxes.conf.cpu()           # (N,)
                out.append(torch.cat([boxes, confs.unsqueeze(1)], dim=1))  # (N, 5)
            else:
                out.append(torch.zeros(0, 5))
        return out

    @torch.no_grad()
    def update(self, enhanced: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            enhanced, gt: (B, 3, H, W) float in [0, 1]
        """
        gt_preds  = self._run_yolo(gt)        # pseudo-GT boxes from bright frame
        enh_preds = self._run_yolo(enhanced)  # predictions from enhanced frame

        for pred, pseudo_gt in zip(enh_preds, gt_preds):
            gt_boxes = pseudo_gt[:, :4] if len(pseudo_gt) > 0 else torch.zeros(0, 4)
            ap = _compute_ap50(pred, gt_boxes)
            self._aps.append(ap)

    def compute(self) -> dict:
        if not self._aps:
            return {"map50": float("nan")}
        return {"map50": float(np.mean(self._aps))}


# ---------------------------------------------------------------------------
# 4. ORB keypoints + homography inlier ratio
# ---------------------------------------------------------------------------
class ORBMetrics:
    """
    SLAM-proxy metrics using ORB feature matching between consecutive frames.

    For each consecutive pair of enhanced frames:
      - keypoint_count: mean ORB keypoints detected per frame (higher → better SLAM)
      - inlier_ratio:   RANSAC homography inliers / total matches (higher → better localization)

    Requires: pip install opencv-python
    """

    def __init__(self, n_features: int = 1000, ransac_thresh: float = 5.0):
        try:
            import cv2 as _cv2
            self._cv2 = _cv2
        except ImportError as exc:
            raise ImportError(
                "ORBMetrics requires OpenCV.\n"
                "  pip install opencv-python\n"
                f"  Original error: {exc}"
            ) from exc

        self.orb           = _cv2.ORB_create(nfeatures=n_features)
        self.bf            = _cv2.BFMatcher(_cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_thresh = ransac_thresh
        self._kp_counts    = []
        self._inlier_ratios = []

    def reset(self):
        self._kp_counts     = []
        self._inlier_ratios = []

    def _to_gray_uint8(self, frame: torch.Tensor) -> np.ndarray:
        """(3, H, W) float [0,1] → (H, W) uint8 grayscale."""
        rgb = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        return self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)

    @torch.no_grad()
    def update(self, clips: torch.Tensor):
        """
        Args:
            clips: (B, 3, D, H, W) enhanced video clips, D ≥ 2
        """
        B, C, D, H, W = clips.shape
        if D < 2:
            return

        for b in range(B):
            for d in range(D - 1):
                gray0 = self._to_gray_uint8(clips[b, :, d])
                gray1 = self._to_gray_uint8(clips[b, :, d + 1])

                kp0, des0 = self.orb.detectAndCompute(gray0, None)
                kp1, des1 = self.orb.detectAndCompute(gray1, None)

                self._kp_counts.append((len(kp0) + len(kp1)) / 2.0)

                if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
                    self._inlier_ratios.append(0.0)
                    continue

                matches = self.bf.match(des0, des1)
                if len(matches) < 4:
                    self._inlier_ratios.append(0.0)
                    continue

                pts0 = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                _, mask = self._cv2.findHomography(
                    pts0, pts1,
                    self._cv2.RANSAC,
                    self.ransac_thresh,
                )
                n_inliers = int(mask.sum()) if mask is not None else 0
                self._inlier_ratios.append(n_inliers / len(matches))

    def compute(self) -> dict:
        if not self._kp_counts:
            return {"orb_keypoints": float("nan"),
                    "homography_inlier_ratio": float("nan")}
        return {
            "orb_keypoints":          float(np.mean(self._kp_counts)),
            "homography_inlier_ratio": float(np.mean(self._inlier_ratios)),
        }


# ---------------------------------------------------------------------------
# 5. Flow consistency (EPE vs gamma-brightened reference)
# ---------------------------------------------------------------------------
class FlowConsistency:
    """
    Optical flow consistency: EPE between enhanced-frame flow and
    gamma-brightened-input flow (gamma=0.4 as a bright proxy).

    For each consecutive pair (t, t+1):
        flow_enh = RAFT(enhanced_t, enhanced_{t+1})
        flow_ref = RAFT(low_t^0.4,  low_{t+1}^0.4)
        EPE      = mean ||flow_enh - flow_ref||_2

    Lower EPE → enhanced flow matches the gamma-reference closely
    (enhancement preserves motion estimation).

    Requires: FrozenRAFT from training.raft_warp (torchvision >= 0.13)
    """

    def __init__(self, device: torch.device, num_flow_updates: int = 12):
        from training.raft_warp import FrozenRAFT
        self.raft   = FrozenRAFT(num_flow_updates=num_flow_updates).to(device)
        self.device = device
        self._epes  = []

    def reset(self):
        self._epes = []

    @torch.no_grad()
    def update(self, enhanced: torch.Tensor, low: torch.Tensor):
        """
        Args:
            enhanced: (B, 3, D, H, W) enhanced clip
            low:      (B, 3, D, H, W) original low-light clip
        """
        D = enhanced.shape[2]
        if D < 2:
            return

        enhanced = enhanced.to(self.device)
        low      = low.to(self.device)

        for d in range(D - 1):
            enh_t0  = enhanced[:, :, d,     :, :]
            enh_t1  = enhanced[:, :, d + 1, :, :]
            ref_t0  = low[:,      :, d,     :, :].pow(0.4)
            ref_t1  = low[:,      :, d + 1, :, :].pow(0.4)

            flow_enh = self.raft(enh_t0, enh_t1)   # (B, 2, H, W)
            flow_ref = self.raft(ref_t0, ref_t1)

            epe = (flow_enh - flow_ref).norm(dim=1).mean().item()
            self._epes.append(epe)

    def compute(self) -> dict:
        if not self._epes:
            return {"flow_epe": float("nan")}
        return {"flow_epe": float(np.mean(self._epes))}
