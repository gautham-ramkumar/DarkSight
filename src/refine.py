import time
import datetime
from pathlib import Path

import torch
import torchvision
from torch.amp import autocast, GradScaler

from core.model             import Zero3DCE
from training.losses        import Zero3DCELoss, PSNR, SSIMLoss
from training.perception_loss import PerceptionLoss
from data.dataloader        import (get_recurrent_train_dataloader, get_image_train_dataloader,
                                    get_val_dataloader, get_video_metric_loader)
from train import train_one_epoch, validate, validate_heavy_metrics, save_checkpoint, CONFIG as TRAIN_CONFIG

# ---------------------------------------------------------------------------
# Refinement Configuration (v2.1 recovery)
# ---------------------------------------------------------------------------
CONFIG = TRAIN_CONFIG.copy()
CONFIG.update({
    "epochs"         : 15,          # Short burst refinement
    "lr"             : 5e-5,        # Fixed LR for stability
    "resume"         : str(Path(__file__).resolve().parent.parent / "checkpoints" / "best.pth"),
    "perception_loss_weight" : 0.5, # Sharpening weights (v2.1)
    "max_batches_per_epoch"  : 2000,
    "metric_interval"        : 5,
})

def refine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Refinement Mode on: {device}")

    # Directories
    run_id       = "refine_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = CONFIG["checkpoint_dir"] / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID    : {run_id}")

    # Model - ensure v2.1 features
    model = Zero3DCE(
        predict_illumination = True,
        use_recurrent        = True,
        tbptt_steps          = 4,
    ).to(device)

    # Losses
    criterion            = Zero3DCELoss().to(device)
    perception_criterion = PerceptionLoss(gamma=0.4).to(device)
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIMLoss().to(device)

    # RAFT (always on for refine)
    from training.raft_warp import FrozenRAFT, WarpTemporalLoss
    raft             = FrozenRAFT().to(device)
    warp_temporal_fn = WarpTemporalLoss().to(device)

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scaler    = GradScaler('cuda')

    # Load the "Blurry Best" but RESET the best_psnr to 0
    print(f"Loading base model for refinement: {CONFIG['resume']}")
    ckpt = torch.load(CONFIG["resume"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    # We ignore optimizer/scheduler state from the old run to start refinement fresh
    best_val_psnr = 0.0 
    print("  [ok] Model loaded. PSNR baseline reset to 0.0 for v2.1 recovery.")

    # Data
    video_loader = get_recurrent_train_dataloader(batch_size=2, seq_len=8, num_workers=4)
    image_loader = get_image_train_dataloader(batch_size=2, num_workers=4)
    val_loader   = get_val_dataloader(batch_size=2, num_workers=4)
    
    # Heavy metrics
    from eval.metrics import YChannelPSNR, LPIPSMetric, DetectionMAP, ORBMetrics, FlowConsistency
    psnr_y_fn        = YChannelPSNR().to(device)
    lpips_fn         = LPIPSMetric().to(device)
    det_map          = DetectionMAP(device=device)
    orb_metrics      = ORBMetrics()
    flow_consistency = FlowConsistency(device=device)
    vid_metric_loader = get_video_metric_loader(batch_size=2)

    print("=" * 60)
    print(f"Starting Refinement Burst: {CONFIG['epochs']} epochs")
    print("=" * 60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.perf_counter()
        
        train_losses = train_one_epoch(
            model, criterion, perception_criterion, optimizer,
            video_loader, image_loader, scaler, device, epoch,
            raft=raft, warp_temporal_fn=warp_temporal_fn,
        )

        val_loss, val_psnr, val_ssim, val_extras = validate(
            model, criterion, psnr_fn, ssim_fn, val_loader, device,
            lpips_fn = lpips_fn, psnr_y_fn = psnr_y_fn
        )

        heavy_metrics = {}
        if epoch % CONFIG["metric_interval"] == 0 or epoch == 1:
            heavy_metrics = validate_heavy_metrics(
                model, val_loader, vid_metric_loader, device,
                det_map, orb_metrics, flow_consistency
            )

        epoch_time = time.perf_counter() - epoch_start
        print(f"\nEpoch {epoch:02d} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f} | LPIPS: {val_extras['val_lpips']:.3f} | Time: {epoch_time:.1f}s")
        if heavy_metrics:
            print(f"  Utility: mAP={heavy_metrics.get('val_map50',0):.3f} | Inliers={heavy_metrics.get('vid_homography_inlier_ratio',0):.3f}")

        # SAVE EVERYTHING in refinement
        state = {
            "epoch": epoch, 
            "model": model.state_dict(), 
            "val_psnr": val_psnr,
            "run_id": run_id
        }
        save_checkpoint(state, run_ckpt_dir / f"refine_epoch_{epoch:03d}.pth")
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_checkpoint(state, run_ckpt_dir / "best_refined.pth")
            print(f"  *** New best refined model saved! ***")

    print("\nRefinement complete. Checkpoints saved in:", run_ckpt_dir)

if __name__ == "__main__":
    refine()
