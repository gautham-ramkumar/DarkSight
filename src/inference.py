"""
Inference script for Zero-3DCE best.pth checkpoint.

Runs on the full LOLv1 + LOLv2 validation sets, reports mean PSNR/SSIM,
and saves side-by-side comparison images to test_outputs/inference/.

Usage:
    python3 inference.py                          # uses checkpoints/best.pth
    python3 inference.py --checkpoint path/to.pth
"""

import argparse
from pathlib import Path

import torch
import torchvision
from torch.amp import autocast

from core.model      import Zero3DCE
from data.dataloader import get_val_dataloader
from training.losses import PSNR, SSIMLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "best.pth"))
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save low / enhanced / GT comparison grids")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = Zero3DCE().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    saved_epoch = ckpt.get("epoch",    "?")
    saved_psnr  = ckpt.get("val_psnr", float("nan"))
    saved_ssim  = ckpt.get("val_ssim", float("nan"))
    run_id      = ckpt.get("run_id",   "legacy")
    print(f"Checkpoint epoch : {saved_epoch}  |  run: {run_id}")
    print(f"Saved val PSNR   : {saved_psnr:.4f} dB")
    print(f"Saved val SSIM   : {saved_ssim:.4f}\n")

    psnr_fn = PSNR().to(device)
    ssim_fn = SSIMLoss().to(device)

    val_loader = get_val_dataloader(
        batch_size  = args.batch_size,
        image_size  = args.image_size,
        num_workers = args.num_workers,
    )

    out_dir = Path(__file__).resolve().parent.parent / "test_outputs" / "inference"
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    n_batches  = len(val_loader)

    with torch.no_grad():
        for i, (low, gt) in enumerate(val_loader):
            low = low.to(device, non_blocking=True)
            gt  = gt.to(device, non_blocking=True)

            with autocast("cuda"):
                _, enhanced, _illum, _ = model(low)

            enh_frame = enhanced[:, :, 0]   # (B, C, H, W)
            gt_frame  = gt[:,     :, 0]

            total_psnr += psnr_fn(enh_frame, gt_frame).item()
            total_ssim += ssim_fn(enh_frame, gt_frame).item()

            if args.save_images and i < 10:
                low_frame = low[:, :, 0]
                grid = torchvision.utils.make_grid(
                    torch.cat([low_frame, enh_frame, gt_frame], dim=0),
                    nrow    = low_frame.shape[0],
                    padding = 2,
                )
                torchvision.utils.save_image(grid, out_dir / f"batch_{i:03d}.png")

    mean_psnr = total_psnr / n_batches
    mean_ssim = total_ssim / n_batches

    print("=" * 45)
    print(f"Val PSNR  : {mean_psnr:.4f} dB")
    print(f"Val SSIM  : {mean_ssim:.4f}")
    print("=" * 45)
    if args.save_images:
        print(f"Images saved to: {out_dir}/")


if __name__ == "__main__":
    main()
