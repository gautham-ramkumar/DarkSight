import csv
import time
import math
import datetime
from pathlib import Path

import torch
import torchvision
from torch.amp import autocast, GradScaler

from core.model             import Zero3DCE
from training.losses        import Zero3DCELoss, PSNR, SSIMLoss
from training.perception_loss import PerceptionLoss
from data.dataloader        import (get_video_train_dataloader, get_image_train_dataloader,
                                    get_val_dataloader, get_recurrent_train_dataloader,
                                    get_video_metric_loader)
# training.raft_warp / eval.metrics are imported lazily below when their CONFIG flags are True

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "checkpoint_dir" : Path(__file__).resolve().parent.parent / "checkpoints",
    "log_dir"        : Path(__file__).resolve().parent.parent / "logs",

    "epochs"         : 100,
    "batch_size"     : 2,    # Reduced for v2 D=8 sequences
    "image_size"     : 256,
    "num_workers"    : 4,

    "lr"             : 1e-4,
    "weight_decay"   : 1e-4,
    "betas"          : (0.9, 0.999),

    "lr_min"         : 1e-6,
    "max_grad_norm"  : 1.0,

    # Set to "auto" to resume from last.pth if it exists, or provide an explicit path.
    # Set to None to always start fresh (checkpoints + log are overwritten).
    "resume"         : "auto",

    # Perception loss (Phase 1) — set weight to 0 to disable without removing the module
    "perception_loss_weight" : 0.5,   # Reduced to allow spatial/edge losses to sharpen (v2.1)
    "perception_w_temporal"  : 1.0,   # split within perception loss
    "perception_w_perceptual": 1.0,
    "perception_gamma"       : 0.4,   # gamma for synthetic bright reference

    # ConvGRU recurrent mode (Phase 2b)
    # Set use_recurrent=True to switch the model to frame-by-frame processing
    # with a ConvGRU bottleneck.  Replaces the D=2 video loader with longer
    # sequences.  Requires more VRAM — reduce batch_size if needed.
    "use_recurrent"       : True,
    "recurrent_seq_len"   : 8,     # frames per training sequence (D for recurrent loader)
    "recurrent_batch_size": 2,     # D=8 clips use ~4× more memory than D=2 pairs
    "tbptt_steps"         : 4,     # detach GRU hidden state every N frames

    # RAFT optical flow (Phase 3) — training-only, zero inference cost.
    # Replaces alpha-map temporal TV with flow-compensated warp loss on video
    # batches (D > 1).  Image batches (D=1) still use alpha TV as fallback.
    # Requires torchvision >= 0.13:  pip install --upgrade torchvision
    "use_raft"            : True,
    "raft_flow_updates"   : 12,    # RAFT-small refinement iterations (12 = best quality)

    # Phase 4 perception metrics — logged to CSV alongside PSNR/SSIM.
    # Lightweight metrics (LPIPS, Y-PSNR) run every epoch.
    # Heavy metrics (detection mAP, ORB, flow EPE) run every metric_interval epochs.
    # Set use_perception_metrics=False to skip all Phase 4 metrics.
    # Dependencies: pip install lpips ultralytics opencv-python
    #               pip install --upgrade torchvision  (for FlowConsistency)
    "use_perception_metrics" : True,
    "metric_interval"        : 5,     # heavy metrics every N epochs
    "metric_max_vid_samples" : 200,   # DarkVideo pairs used for ORB / flow EPE
    "max_batches_per_epoch"  : 2500,  # Limits epoch length for faster feedback
}

# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
class CSVLogger:
    COLUMNS = [
        "epoch", "lr",
        "train_total", "train_color", "train_exposure", "train_spatial",
        "train_smooth_spa", "train_smooth_temp", "train_edge", "train_ms_ssim",
        "train_perception_temporal", "train_perception_perceptual",
        "train_warp_temporal",
        "val_total", "val_psnr", "val_ssim",
        # Phase 4 — single-frame (every epoch when use_perception_metrics=True)
        "val_psnr_y", "val_lpips",
        # Phase 4 — heavy (every metric_interval epochs)
        "val_map50",
        "vid_orb_keypoints", "vid_homography_inlier_ratio", "vid_flow_epe",
        "epoch_time_sec",
    ]

    def __init__(self, path: Path):
        """Append if exists, otherwise create fresh with header."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writerow(row)

# ---------------------------------------------------------------------------
# Training loop — one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, criterion, perception_criterion, optimizer,
                    video_loader, image_loader, scaler, device, epoch,
                    raft=None, warp_temporal_fn=None):
    """
    One training epoch over two loaders:
      - video_loader: DarkVideo D=2 consecutive pairs
      - image_loader: LOLv1+LOLv2 low images as D=1 clips
    Both loaders are iterated in full; losses are averaged over all batches.

    When raft and warp_temporal_fn are provided (use_raft=True in CONFIG),
    video batches (D > 1) use flow-compensated temporal loss instead of
    alpha-map temporal TV.  Image batches (D=1) fall back to alpha TV.
    """
    model.train()
    running   = {}
    n_video   = len(video_loader)
    n_image   = len(image_loader)
    n_batches = n_video + n_image
    p_weight  = CONFIG["perception_loss_weight"]
    max_b     = CONFIG.get("max_batches_per_epoch", float('inf'))
    total_it  = 0

    for loader, label, n in [(video_loader, "vid", n_video),
                              (image_loader, "img", n_image)]:
        for batch_idx, (low, _) in enumerate(loader):
            if total_it >= max_b:
                break
            
            low = low.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # RAFT runs FP32 internally and is @no_grad; compute flows before autocast
            flows = None
            if raft is not None and low.shape[2] > 1:
                flows = raft.estimate_consecutive_flows(low) or None

            with autocast('cuda'):
                A_maps, enhanced, _illum, _hidden = model(low)

                # Warp loss is differentiable through `enhanced`; flows are detached
                warp_loss = warp_temporal_fn(enhanced, flows) if flows else None

                total, loss_dict                  = criterion(A_maps, enhanced, low,
                                                              warp_temporal=warp_loss)
                perc_total, perc_dict             = perception_criterion(enhanced, low)
                combined = total + p_weight * perc_total

            if not torch.isfinite(combined):
                print(f"  [warn] Non-finite loss at epoch {epoch} "
                      f"[{label} {batch_idx+1}] — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(combined).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v
            for k, v in perc_dict.items():
                running[k] = running.get(k, 0.0) + v
            if warp_loss is not None:
                running["warp_temporal"] = running.get("warp_temporal", 0.0) + warp_loss.item()

            total_it += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n or total_it == max_b:
                warp_str = f"  warp: {warp_loss.item():>7.4f}" if warp_loss is not None else ""
                print(f"  Epoch {epoch:03d} [{label} {batch_idx+1:03d}/{n}] "
                      f"loss: {combined.item():>7.4f}  "
                      f"zero_ref: {loss_dict['total']:>7.4f}  "
                      f"perc: {perc_dict['perception_total']:>7.4f}"
                      f"{warp_str}")

    avg = {f"train_{k}": v / total_it for k, v in running.items()}
    return avg

# ---------------------------------------------------------------------------
# Validation loop — one epoch
# ---------------------------------------------------------------------------
def validate(model, criterion, psnr_fn, ssim_fn, loader, device,
             debug_dir: Path = None, lpips_fn=None, psnr_y_fn=None):
    """
    Val set uses D=1 clips — PSNR/SSIM computed on frame 0 (B, C, H, W),
    directly comparable to standard LOL benchmark results.

    If lpips_fn / psnr_y_fn are provided (Phase 4), LPIPS and Y-channel PSNR
    are computed in the same pass at no extra model-forward cost.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips  = 0.0
    total_psnr_y = 0.0
    n_batches    = len(loader)
    saved_debug  = False

    with torch.no_grad():
        for low, gt in loader:
            low = low.to(device, non_blocking=True)
            gt  = gt.to(device, non_blocking=True)

            with autocast('cuda'):
                A_maps, enhanced, _illum, _hidden = model(low)
                loss, _ = criterion(A_maps, enhanced, low)

            total_loss += loss.item()

            # Compare frame 0 only: (B, C, H, W) — gives true per-image PSNR
            enh_frame = enhanced[:, :, 0, :, :]   # (B, C, H, W)
            gt_frame  = gt[:,       :, 0, :, :]

            total_psnr += psnr_fn(enh_frame, gt_frame).item()
            total_ssim += ssim_fn(enh_frame, gt_frame).item()

            if psnr_y_fn is not None:
                total_psnr_y += psnr_y_fn(enh_frame, gt_frame).item()
            if lpips_fn is not None:
                total_lpips += lpips_fn(enh_frame.float(), gt_frame.float()).item()

            if debug_dir is not None and not saved_debug:
                debug_dir.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(low[:1, :, 0],      debug_dir / "debug_low.png")
                torchvision.utils.save_image(enh_frame[:1],      debug_dir / "debug_enhanced.png")
                torchvision.utils.save_image(gt_frame[:1],       debug_dir / "debug_gt.png")
                print(f"  [debug] Saved sample images to {debug_dir}/")
                saved_debug = True

    extras = {}
    if psnr_y_fn is not None:
        extras["val_psnr_y"] = total_psnr_y / n_batches
    if lpips_fn is not None:
        extras["val_lpips"] = total_lpips / n_batches

    return (total_loss / n_batches,
            total_psnr / n_batches,
            total_ssim / n_batches,
            extras)

# ---------------------------------------------------------------------------
# Heavy perception metrics (detection mAP, ORB, flow EPE)
# ---------------------------------------------------------------------------
def validate_heavy_metrics(model, val_loader, vid_loader, device,
                            det_map, orb_metrics, flow_consistency):
    """
    Compute heavy Phase 4 metrics.  Called every metric_interval epochs.

    val_loader  — LOL val set (single frames) for detection mAP
    vid_loader  — DarkVideo clips (D=2 pairs) for ORB + flow EPE
    det_map / orb_metrics / flow_consistency — metric objects (may be None)
    """
    model.eval()
    result = {}

    # Detection mAP (single frames, LOL val)
    if det_map is not None:
        det_map.reset()
        with torch.no_grad():
            for low, gt in val_loader:
                low = low.to(device, non_blocking=True)
                gt  = gt.to(device, non_blocking=True)
                with autocast('cuda'):
                    _, enhanced, _illum, _hidden = model(low)
                enh_frame = enhanced[:, :, 0, :, :]
                gt_frame  = gt[:,       :, 0, :, :]
                det_map.update(enh_frame, gt_frame)
        result.update({f"val_{k}": v for k, v in det_map.compute().items()})

    # ORB + flow EPE (video clips, DarkVideo)
    if vid_loader is not None and (orb_metrics is not None or flow_consistency is not None):
        if orb_metrics is not None:
            orb_metrics.reset()
        if flow_consistency is not None:
            flow_consistency.reset()

        with torch.no_grad():
            for low_clip, _ in vid_loader:
                low_clip = low_clip.to(device, non_blocking=True)
                with autocast('cuda'):
                    _, enhanced_clip, _illum, _hidden = model(low_clip)
                if orb_metrics is not None:
                    orb_metrics.update(enhanced_clip)
                if flow_consistency is not None:
                    flow_consistency.update(enhanced_clip, low_clip)

        if orb_metrics is not None:
            result.update({f"vid_{k}": v for k, v in orb_metrics.compute().items()})
        if flow_consistency is not None:
            result.update({f"vid_{k}": v for k, v in flow_consistency.compute().items()})

    return result


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_psnr = ckpt.get("best_val_psnr", 0.0)
    print(f"  Resumed at epoch {start_epoch}, best val PSNR: {best_val_psnr:.2f} dB")
    return start_epoch, best_val_psnr

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)

    run_id       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = CONFIG["checkpoint_dir"] / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID    : {run_id}")
    print(f"Checkpoints → {run_ckpt_dir}  (also mirrored to {CONFIG['checkpoint_dir']})\n")

    model = Zero3DCE(
        predict_illumination = True,  # Enabled for v2
        use_recurrent        = CONFIG["use_recurrent"],
        tbptt_steps          = CONFIG["tbptt_steps"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    criterion            = Zero3DCELoss().to(device)
    perception_criterion = PerceptionLoss(
        w_temporal   = CONFIG["perception_w_temporal"],
        w_perceptual = CONFIG["perception_w_perceptual"],
        gamma        = CONFIG["perception_gamma"],
    ).to(device)
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIMLoss().to(device)

    raft             = None
    warp_temporal_fn = None
    if CONFIG["use_raft"]:
        from training.raft_warp import FrozenRAFT, WarpTemporalLoss
        print("Loading RAFT-small (frozen, training-only) ...")
        raft             = FrozenRAFT(num_flow_updates=CONFIG["raft_flow_updates"]).to(device)
        warp_temporal_fn = WarpTemporalLoss().to(device)
        n_raft_params    = sum(p.numel() for p in raft.parameters())
        print(f"  RAFT-small parameters: {n_raft_params:,}  (frozen — zero gradient cost)\n")

    # ---- Phase 4 metrics ---------------------------------------------------
    lpips_fn        = None
    psnr_y_fn       = None
    det_map         = None
    orb_metrics     = None
    flow_consistency = None
    vid_metric_loader = None

    if CONFIG["use_perception_metrics"]:
        from eval.metrics import YChannelPSNR, LPIPSMetric, DetectionMAP, ORBMetrics, FlowConsistency
        print("Initialising Phase 4 perception metrics ...")
        psnr_y_fn = YChannelPSNR().to(device)

        try:
            lpips_fn = LPIPSMetric().to(device)
            print("  [ok] LPIPS (AlexNet)")
        except ImportError as e:
            print(f"  [skip] LPIPS: {e}")

        try:
            det_map = DetectionMAP(device=device)
            print("  [ok] Detection mAP (YOLOv8n)")
        except ImportError as e:
            print(f"  [skip] Detection mAP: {e}")

        try:
            orb_metrics = ORBMetrics()
            print("  [ok] ORB feature metrics")
        except ImportError as e:
            print(f"  [skip] ORB: {e}")

        try:
            flow_consistency = FlowConsistency(device=device,
                                               num_flow_updates=CONFIG["raft_flow_updates"])
            print("  [ok] Flow consistency (RAFT EPE)")
        except ImportError as e:
            print(f"  [skip] Flow consistency: {e}")

        if orb_metrics is not None or flow_consistency is not None:
            print("Loading video metric loader ...")
            vid_metric_loader = get_video_metric_loader(
                batch_size   = CONFIG["batch_size"],
                image_size   = CONFIG["image_size"],
                num_workers  = CONFIG["num_workers"],
                max_samples  = CONFIG["metric_max_vid_samples"],
            )
        print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = CONFIG["lr"],
        betas = CONFIG["betas"],
        weight_decay = CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = CONFIG["epochs"],
        eta_min = CONFIG["lr_min"],
    )

    scaler = GradScaler('cuda')

    print("Loading datasets ...")
    if CONFIG["use_recurrent"]:
        video_loader = get_recurrent_train_dataloader(
            batch_size  = CONFIG["recurrent_batch_size"],
            seq_len     = CONFIG["recurrent_seq_len"],
            image_size  = CONFIG["image_size"],
            num_workers = CONFIG["num_workers"],
        )
        print(f"  [recurrent mode] D={CONFIG['recurrent_seq_len']} sequences, "
              f"batch={CONFIG['recurrent_batch_size']}, tbptt={CONFIG['tbptt_steps']}")
    else:
        video_loader = get_video_train_dataloader(
            batch_size  = CONFIG["batch_size"],
            image_size  = CONFIG["image_size"],
            num_workers = CONFIG["num_workers"],
        )

    image_loader = get_image_train_dataloader(
        batch_size  = CONFIG["batch_size"],
        image_size  = CONFIG["image_size"],
        num_workers = CONFIG["num_workers"],
    )
    val_loader = get_val_dataloader(
        batch_size  = CONFIG["batch_size"],
        image_size  = CONFIG["image_size"],
        num_workers = CONFIG["num_workers"],
    )

    log_path = CONFIG["log_dir"] / "training_log.csv"
    # CSVLogger always overwrites the file with a fresh header on init.
    # Old checkpoints (best.pth / last.pth) are overwritten per-epoch too.
    logger = CSVLogger(log_path)
    print(f"Logging to: {log_path}  (overwritten fresh this run)\n")

    start_epoch = 1
    best_val_psnr = 0.0

    resume_cfg = CONFIG["resume"]
    if resume_cfg == "auto":
        # Automatically continue from the last saved checkpoint if available
        resume_path = CONFIG["checkpoint_dir"] / "last.pth"
        if resume_path.exists():
            start_epoch, best_val_psnr = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
        else:
            print("No last.pth found — starting fresh.")
    elif resume_cfg is not None:
        resume_path = Path(resume_cfg)
        if resume_path.exists():
            start_epoch, best_val_psnr = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
        else:
            print(f"Warning: checkpoint not found at {resume_path} — starting fresh.")

    print("=" * 60)
    print(f"Starting Zero-3DCE training: {CONFIG['epochs']} epochs")
    print(f"Video batches/epoch : {len(video_loader)}")
    print(f"Image batches/epoch : {len(image_loader)}")
    print(f"Val   batches/epoch : {len(val_loader)}")
    print("=" * 60)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:03d}/{CONFIG['epochs']}  LR: {current_lr:.2e}")
        print("-" * 50)

        train_losses = train_one_epoch(
            model, criterion, perception_criterion, optimizer,
            video_loader, image_loader, scaler, device, epoch,
            raft=raft, warp_temporal_fn=warp_temporal_fn,
        )

        debug_dir = CONFIG["log_dir"].parent / "test_outputs" / "debug" / f"epoch_{epoch:03d}" if epoch in (1, 5, 10, 20) else None
        val_loss, val_psnr, val_ssim, val_extras = validate(
            model, criterion, psnr_fn, ssim_fn, val_loader, device,
            debug_dir  = debug_dir,
            lpips_fn   = lpips_fn,
            psnr_y_fn  = psnr_y_fn,
        )

        # Heavy metrics every metric_interval epochs
        heavy_metrics = {}
        if (CONFIG["use_perception_metrics"] and
                epoch % CONFIG["metric_interval"] == 0):
            print("  [metrics] Running heavy perception metrics ...")
            heavy_metrics = validate_heavy_metrics(
                model, val_loader, vid_metric_loader, device,
                det_map, orb_metrics, flow_consistency,
            )

        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        print(f"\n  Train loss : {train_losses['train_total']:.4f}")
        print(f"  Val   loss : {val_loss:.4f}")
        print(f"  Val   PSNR : {val_psnr:.2f} dB  (RGB)")
        if "val_psnr_y" in val_extras:
            print(f"  Val   PSNR : {val_extras['val_psnr_y']:.2f} dB  (Y-channel)")
        if "val_lpips" in val_extras:
            print(f"  Val   LPIPS: {val_extras['val_lpips']:.4f}")
        print(f"  Val   SSIM : {val_ssim:.4f}")
        if heavy_metrics:
            if "val_map50" in heavy_metrics:
                print(f"  Det  mAP50 : {heavy_metrics['val_map50']:.4f}")
            if "vid_orb_keypoints" in heavy_metrics:
                print(f"  ORB  kpts  : {heavy_metrics['vid_orb_keypoints']:.1f}")
            if "vid_homography_inlier_ratio" in heavy_metrics:
                print(f"  Homo inlier: {heavy_metrics['vid_homography_inlier_ratio']:.3f}")
            if "vid_flow_epe" in heavy_metrics:
                print(f"  Flow EPE   : {heavy_metrics['vid_flow_epe']:.4f}")
        print(f"  Time       : {epoch_time:.1f}s")

        state = {
            "epoch"         : epoch,
            "run_id"        : run_id,
            "model"         : model.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            "scaler"        : scaler.state_dict(),
            "best_val_psnr" : best_val_psnr,
            "val_psnr"      : val_psnr,
            "val_ssim"      : val_ssim,
        }

        if val_psnr > best_val_psnr:
            best_val_psnr   = val_psnr
            state["best_val_psnr"] = best_val_psnr
            # Versioned copy inside the run dir + backward-compat copy at root
            save_checkpoint(state, run_ckpt_dir / "best.pth")
            save_checkpoint(state, CONFIG["checkpoint_dir"] / "best.pth")
            print(f"  Saved best checkpoint (val PSNR: {best_val_psnr:.2f} dB)")

        # last.pth in both locations for --resume auto
        save_checkpoint(state, run_ckpt_dir / "last.pth")
        save_checkpoint(state, CONFIG["checkpoint_dir"] / "last.pth")

        log_row = {
            "epoch"           : epoch,
            "lr"              : current_lr,
            "val_total"       : val_loss,
            "val_psnr"        : val_psnr,
            "val_ssim"        : val_ssim,
            "epoch_time_sec"  : epoch_time,
            **train_losses,
            **val_extras,
            **heavy_metrics,
        }
        logger.log(log_row)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best val PSNR : {best_val_psnr:.2f} dB")
    print("=" * 60)

if __name__ == "__main__":
    train()