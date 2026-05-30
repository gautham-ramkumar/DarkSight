"""
TensorRT inference for Zero3DCE.

Supports three backends:
    1. tensorrt   — full TRT engine (.trt file), fastest
    2. onnxruntime — ONNX model via ORT-GPU, easy fallback
    3. pytorch    — original PyTorch model, reference baseline

Usage:
    # TRT engine (requires tensorrt + built engine):
    python3 trt_infer.py --backend trt --engine ../exports/zero3dce_fp16.trt

    # ONNX Runtime (no TRT install needed):
    python3 trt_infer.py --backend onnx --onnx ../exports/zero3dce.onnx

    # PyTorch baseline:
    python3 trt_infer.py --backend pytorch --checkpoint ../checkpoints/best.pth

Install:
    pip install onnx onnxruntime-gpu
    pip install tensorrt          # optional, for --backend trt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataloader import get_val_dataloader
from training.losses import PSNR, SSIMLoss
from core.model import Zero3DCE


# ---------------------------------------------------------------------------
# Backend wrappers — uniform .run(input_tensor) → enhanced_tensor interface
# ---------------------------------------------------------------------------
class PyTorchBackend:
    def __init__(self, checkpoint: str | None, device: torch.device):
        self.device = device

        has_illum = False
        has_gru   = False

        if checkpoint:
            ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt)
            has_illum = any(k.startswith("illum_head") for k in state)
            has_gru   = any(k.startswith("gru")        for k in state)
            run_id    = ckpt.get("run_id", "legacy")
            print(f"[PyTorch] Loaded checkpoint: {checkpoint}")
            print(f"[PyTorch]   run={run_id}  has_illum={has_illum}  has_gru={has_gru}")
        else:
            state = None
            print("[PyTorch] No checkpoint — random weights (baseline test).")

        self.model = Zero3DCE(
            predict_illumination = has_illum,
            use_recurrent        = has_gru,
        ).to(device)

        if state is not None:
            self.model.load_state_dict(state)

        self.model.eval()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, enhanced, _illum, _hidden = self.model(x)
        return enhanced

    def name(self) -> str:
        return "PyTorch"


class ONNXBackend:
    def __init__(self, onnx_path: str, device: torch.device):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda" and
               "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.sess      = ort.InferenceSession(onnx_path, providers=providers)
        self.device    = device
        self._provider = providers[0].replace("ExecutionProvider", "")
        print(f"[ONNX] Session: {onnx_path}  provider={self._provider}")

    def run(self, x: torch.Tensor) -> torch.Tensor:
        x_np    = x.cpu().numpy()
        outputs = self.sess.run(None, {"input": x_np})
        # outputs[0] = alpha_maps, outputs[1] = enhanced
        return torch.from_numpy(outputs[1]).to(self.device)

    def name(self) -> str:
        return f"ONNX-{self._provider}"


class TRTBackend:
    """
    TensorRT FP16 engine inference via cuda-python bindings.

    Requires:
        pip install tensorrt cuda-python
    """
    def __init__(self, engine_path: str, device: torch.device):
        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as e:
            raise ImportError(
                f"TRT backend requires tensorrt + cuda-python: {e}\n"
                "  pip install tensorrt cuda-python"
            ) from e

        self.device = device
        TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)
        runtime     = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded: {engine_path}")

        self._allocate_buffers()

    def _allocate_buffers(self):
        import tensorrt as trt
        from cuda import cudart

        self._host_inputs  = {}
        self._host_outputs = {}
        self._dev_inputs   = {}
        self._dev_outputs  = {}

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize

            h_buf = np.empty(shape, dtype=dtype)
            err, d_buf = cudart.cudaMalloc(size)
            assert err == cudart.cudaError_t.cudaSuccess

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._host_inputs[name]  = h_buf
                self._dev_inputs[name]   = d_buf
            else:
                self._host_outputs[name] = h_buf
                self._dev_outputs[name]  = d_buf

    def run(self, x: torch.Tensor) -> torch.Tensor:
        from cuda import cudart

        stream = torch.cuda.current_stream().cuda_stream

        inp_name = "input"
        h_in = self._host_inputs[inp_name]
        np.copyto(h_in, x.cpu().numpy())
        cudart.cudaMemcpyAsync(
            self._dev_inputs[inp_name], h_in.ctypes.data,
            h_in.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
        )
        self.context.set_tensor_address(inp_name, self._dev_inputs[inp_name])
        for name, d in self._dev_outputs.items():
            self.context.set_tensor_address(name, d)

        self.context.execute_async_v3(stream)

        out_name = "enhanced"
        h_out = self._host_outputs[out_name]
        cudart.cudaMemcpyAsync(
            h_out.ctypes.data, self._dev_outputs[out_name],
            h_out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
        )
        torch.cuda.current_stream().synchronize()

        return torch.from_numpy(h_out.copy()).to(self.device)

    def name(self) -> str:
        return "TensorRT"


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(backend, val_loader, psnr_fn, ssim_fn, device,
             save_images: bool, out_dir: Path, n_save: int = 10):
    total_psnr = 0.0
    total_ssim = 0.0
    total_ms   = 0.0
    n_batches  = len(val_loader)

    if save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, (low, gt) in enumerate(val_loader):
        low = low.to(device, non_blocking=True)
        gt  = gt.to(device,  non_blocking=True)

        t0       = time.perf_counter()
        enhanced = backend.run(low)
        torch.cuda.synchronize() if device.type == "cuda" else None
        total_ms += (time.perf_counter() - t0) * 1000

        enh_frame = enhanced[:, :, 0]   # (B, C, H, W) from (B, C, D, H, W)
        gt_frame  = gt[:,     :, 0]

        total_psnr += psnr_fn(enh_frame, gt_frame).item()
        total_ssim += ssim_fn(enh_frame, gt_frame).item()

        if save_images and i < n_save:
            low_frame = low[:, :, 0]
            grid = torchvision.utils.make_grid(
                torch.cat([low_frame, enh_frame, gt_frame], dim=0),
                nrow=low_frame.shape[0], padding=2,
            )
            torchvision.utils.save_image(grid, out_dir / f"batch_{i:03d}.png")

    mean_psnr = total_psnr / n_batches
    mean_ssim = total_ssim / n_batches
    mean_ms   = total_ms   / n_batches
    depth     = 2  # D dimension used for FPS
    fps       = depth / (mean_ms / 1000)

    return mean_psnr, mean_ssim, mean_ms, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--backend", choices=["pytorch", "onnx", "trt"],
                        default="onnx")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch .pth checkpoint (--backend pytorch)")
    parser.add_argument("--onnx", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent /
                                    "exports" / "zero3dce.onnx"),
                        help="ONNX model path (--backend onnx)")
    parser.add_argument("--engine", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent /
                                    "exports" / "zero3dce_fp16.trt"),
                        help="TRT engine path (--backend trt)")

    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--output_dir",  type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent /
                                    "test_outputs" / "trt_infer"))

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    if args.backend == "pytorch":
        backend = PyTorchBackend(args.checkpoint, device)
    elif args.backend == "onnx":
        backend = ONNXBackend(args.onnx, device)
    elif args.backend == "trt":
        backend = TRTBackend(args.engine, device)

    print(f"Backend: {backend.name()}\n")

    psnr_fn = PSNR().to(device)
    ssim_fn = SSIMLoss().to(device)

    val_loader = get_val_dataloader(
        batch_size  = args.batch_size,
        image_size  = args.image_size,
        num_workers = args.num_workers,
    )
    print(f"Val batches: {len(val_loader)}\n")

    out_dir = Path(args.output_dir)
    psnr, ssim, ms, fps = evaluate(
        backend, val_loader, psnr_fn, ssim_fn, device,
        save_images=args.save_images, out_dir=out_dir,
    )

    print("=" * 50)
    print(f"Backend   : {backend.name()}")
    print(f"Val PSNR  : {psnr:.4f} dB")
    print(f"Val SSIM  : {ssim:.4f}")
    print(f"Latency   : {ms:.2f} ms/batch")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Real-time : {'YES' if fps >= 30 else 'NO'} (≥30 FPS threshold)")
    print("=" * 50)
    if args.save_images:
        print(f"Images saved to: {out_dir}/")


if __name__ == "__main__":
    main()
