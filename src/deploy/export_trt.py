"""
Export Zero3DCE to ONNX and (optionally) TensorRT.

Pipeline:
    PyTorch checkpoint  →  ONNX  →  TensorRT FP16 engine

Usage:
    # ONNX only (no checkpoint needed for pipeline testing)
    python3 export_trt.py

    # ONNX only with trained checkpoint
    python3 export_trt.py --checkpoint ../checkpoints/best.pth

    # ONNX + TRT (requires: pip install tensorrt)
    python3 export_trt.py --checkpoint ../checkpoints/best.pth --trt --fp16

    # Recurrent (ConvGRU) export — hidden state as explicit I/O:
    python3 export_trt.py --checkpoint ../checkpoints/best.pth --recurrent

    # trtexec alternative (after ONNX export):
    #   trtexec --onnx=../exports/zero3dce.onnx \
    #           --saveEngine=../exports/zero3dce_fp16.trt \
    #           --fp16 --workspace=4096

Requirements:
    pip install onnx onnxruntime-gpu
    # For TRT: pip install tensorrt  (or install NVIDIA TensorRT SDK)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.model import Zero3DCE


# ---------------------------------------------------------------------------
# Export wrappers — strip outputs that ONNX cannot trace (None values)
# ---------------------------------------------------------------------------
class _ExportWrapper(nn.Module):
    """Batch-mode export: 4-tuple → (alpha_maps, enhanced, illum)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        A, enhanced, illum, _hidden = self.model(x)
        return A, enhanced, illum


class _RecurrentExportWrapper(nn.Module):
    """
    Recurrent export: takes (x, h_in) and returns (alpha_maps, enhanced, illum, h_out).
    h_in/h_out shape: (B, 32, H//8, W//8)
    The ONNX graph is stateless — the caller threads hidden state between calls.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, h_in: torch.Tensor):
        A, enhanced, illum, h_out = self.model(x, h_in)
        return A, enhanced, illum, h_out


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(wrapper: nn.Module, inputs: tuple, out_path: Path,
                opset: int = 17, recurrent: bool = False) -> Path:
    import onnx

    if recurrent:
        input_names  = ["input", "hidden_in"]
        output_names = ["alpha_maps", "enhanced", "illum", "hidden_out"]
        dynamic_axes = {
            "input":      {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "hidden_in":  {0: "batch", 2: "hidden_h", 3: "hidden_w"},
            "alpha_maps": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "enhanced":   {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "illum":      {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "hidden_out": {0: "batch", 2: "hidden_h", 3: "hidden_w"},
        }
    else:
        input_names  = ["input"]
        output_names = ["alpha_maps", "enhanced", "illum"]
        dynamic_axes = {
            "input":      {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "alpha_maps": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "enhanced":   {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "illum":      {0: "batch", 2: "depth", 3: "height", 4: "width"},
        }

    torch.onnx.export(
        wrapper,
        inputs,
        str(out_path),
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = input_names,
        output_names        = output_names,
        dynamic_axes        = dynamic_axes,
        dynamo              = False
    )

    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)

    size_kb = out_path.stat().st_size / 1024
    print(f"[ONNX] Saved & validated: {out_path}  ({size_kb:.1f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# TensorRT conversion
# ---------------------------------------------------------------------------
def export_trt(onnx_path: Path, trt_path: Path,
               fp16: bool = True, workspace_gb: int = 4) -> Path | None:
    try:
        import tensorrt as trt
    except ImportError:
        print("\n[TRT] tensorrt package not found.")
        print("      Install: pip install tensorrt")
        print("      Or use trtexec CLI:")
        fp16_flag = "--fp16" if fp16 else ""
        print(f"        trtexec --onnx={onnx_path} "
              f"--saveEngine={trt_path} {fp16_flag} "
              f"--workspace={workspace_gb * 1024}")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error [{i}]: {parser.get_error(i).desc()}")
        raise RuntimeError("ONNX → TRT parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 precision enabled")
    else:
        print("[TRT] FP32 precision (FP16 not available or not requested)")

    print("[TRT] Building engine (this may take a few minutes)...")
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed")
    dt = time.perf_counter() - t0
    print(f"[TRT] Build time: {dt:.1f}s")

    with open(trt_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = trt_path.stat().st_size / (1024 ** 2)
    print(f"[TRT] Engine saved: {trt_path}  ({size_mb:.1f} MB)")
    return trt_path


# ---------------------------------------------------------------------------
# Latency benchmark (PyTorch vs ONNX Runtime vs TRT)
# ---------------------------------------------------------------------------
def benchmark_pytorch(wrapper: nn.Module, inputs: tuple | torch.Tensor,
                      label: str = "", n_warmup: int = 10, n_runs: int = 200) -> float:
    dummy = inputs[0] if isinstance(inputs, tuple) else inputs
    device = dummy.device
    for _ in range(n_warmup):
        with torch.no_grad():
            wrapper(*inputs) if isinstance(inputs, tuple) else wrapper(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            wrapper(*inputs) if isinstance(inputs, tuple) else wrapper(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    depth = dummy.shape[2]
    fps   = (n_runs * depth) / dt
    ms    = dt / n_runs * 1000
    tag   = f"  [{label}]" if label else ""
    print(f"[PyTorch]{tag}  {ms:.2f} ms/batch  →  {fps:.1f} FPS  "
          f"(D={depth}, {dummy.shape[-2]}×{dummy.shape[-1]})")
    return fps


def benchmark_onnx(onnx_path: Path, dummy_inputs: list, n_warmup: int = 10,
                   n_runs: int = 200, depth: int = 2) -> float:
    import onnxruntime as ort
    import numpy as np

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = [inp.name for inp in sess.get_inputs()]
    feeds = {name: arr for name, arr in zip(input_names, dummy_inputs)}

    for _ in range(n_warmup):
        sess.run(None, feeds)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feeds)
    dt = time.perf_counter() - t0

    fps = (n_runs * depth) / dt
    ms  = dt / n_runs * 1000
    provider_str = providers[0].replace("ExecutionProvider", "")
    print(f"[ONNX-RT ({provider_str})]  {ms:.2f} ms/batch  →  {fps:.1f} FPS  (D={depth})")
    return fps


def benchmark_trt(trt_path: Path, dummy: torch.Tensor,
                  n_warmup: int = 10, n_runs: int = 200) -> float | None:
    try:
        import tensorrt as trt   # noqa: F401
    except ImportError:
        print("[TRT] Skipping benchmark (tensorrt not installed)")
        return None

    print("[TRT] Benchmark: use trt_infer.py after engine is built.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint",  type=str,  default=None,
                        help="Path to .pth checkpoint (skip for random-weight pipeline test)")
    parser.add_argument("--output_dir",  type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "exports"))
    parser.add_argument("--batch_size",  type=int,  default=1)
    parser.add_argument("--depth",       type=int,  default=2,
                        help="Temporal dimension D of the input clip")
    parser.add_argument("--image_size",  type=int,  default=256)
    parser.add_argument("--opset",       type=int,  default=18)
    parser.add_argument("--trt",         action="store_true",
                        help="Convert the ONNX model to a TensorRT engine")
    parser.add_argument("--fp16",        action="store_true", default=True,
                        help="Use FP16 precision for TRT engine")
    parser.add_argument("--workspace_gb", type=int, default=4,
                        help="TRT builder workspace in GB")
    parser.add_argument("--bench",       action="store_true", default=True,
                        help="Run PyTorch + ONNX latency benchmarks")
    parser.add_argument("--recurrent",   action="store_true",
                        help="Export recurrent (ConvGRU) model with explicit hidden I/O")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model --------------------------------------------------------
    has_illum = False
    has_gru   = False
    run_id    = "unknown"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        has_illum = any(k.startswith("illum_head") for k in state)
        has_gru   = any(k.startswith("gru")        for k in state)
        run_id    = ckpt.get("run_id", "legacy")
        epoch     = ckpt.get("epoch", "?")
        val_psnr  = ckpt.get("val_psnr", float("nan"))
        print(f"Loaded checkpoint : {ckpt_path}")
        print(f"  run={run_id}  epoch={epoch}  val_psnr={val_psnr:.4f} dB")
        print(f"  has_illum_head={has_illum}  has_gru={has_gru}")
    else:
        print("No checkpoint — exporting with random weights (pipeline test).")

    use_recurrent = args.recurrent or has_gru
    model = Zero3DCE(
        predict_illumination = has_illum,
        use_recurrent        = use_recurrent,
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(state)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters  : {total_params:,}")
    print(f"Recurrent export  : {use_recurrent}\n")

    # ---- Build wrapper & dummy inputs -------------------------------------
    B, C, D, H, W = args.batch_size, 3, args.depth, args.image_size, args.image_size

    if use_recurrent:
        wrapper = _RecurrentExportWrapper(model).eval()
        hidden_h, hidden_w = H // 8, W // 8
        # Recurrent model is called one frame at a time (D=1 per call).
        # Exporting with D>1 unrolls the Python loop at trace time, baking in
        # a fixed number of iterations that breaks when the camera sends D=1.
        dummy_x = torch.rand(B, C, 1, H, W, device=device)
        dummy_h = torch.zeros(B, 32, hidden_h, hidden_w, device=device)
        inputs  = (dummy_x, dummy_h)
        onnx_name = "zero3dce_recurrent.onnx"
    else:
        wrapper = _ExportWrapper(model).eval()
        dummy_x = torch.rand(B, C, D, H, W, device=device)
        inputs  = (dummy_x,)
        onnx_name = "zero3dce.onnx"

    print(f"Input shape       : {tuple(dummy_x.shape)}  (B C D H W)")
    if use_recurrent:
        print(f"Hidden shape      : {tuple(dummy_h.shape)}  (B CH Hb Wb)\n")
    else:
        print()

    # ---- ONNX export -------------------------------------------------------
    onnx_path = out_dir / onnx_name
    export_onnx(wrapper, inputs, onnx_path, opset=args.opset, recurrent=use_recurrent)

    # ---- Benchmark ---------------------------------------------------------
    if args.bench:
        print()
        import numpy as np
        dummy_np_list = [t.cpu().numpy() for t in inputs]
        benchmark_pytorch(wrapper, inputs, label="256×256")
        benchmark_onnx(onnx_path, dummy_np_list, depth=D)

        # 1080p PyTorch latency
        print()
        H1080, W1080 = 1080, 1920
        # pad to align=8 (1080 is divisible by 8, 1920 is divisible by 8)
        dummy_1080 = torch.rand(1, C, 1, H1080, W1080, device=device)
        if use_recurrent:
            dummy_h_1080 = torch.zeros(1, 32, H1080 // 8, W1080 // 8, device=device)
            inputs_1080  = (dummy_1080, dummy_h_1080)
        else:
            inputs_1080 = (dummy_1080,)
        benchmark_pytorch(wrapper, inputs_1080, label="1080p", n_warmup=5, n_runs=50)

    # ---- TRT conversion ----------------------------------------------------
    if args.trt:
        print()
        suffix   = "_recurrent" if use_recurrent else ""
        name     = f"zero3dce{suffix}_fp16.trt" if args.fp16 else f"zero3dce{suffix}_fp32.trt"
        trt_path = out_dir / name
        export_trt(onnx_path, trt_path, fp16=args.fp16,
                   workspace_gb=args.workspace_gb)

    print("\nDone.")
    print(f"Artifacts in: {out_dir}/")
    if not args.trt:
        print("Tip: add --trt to also build the TensorRT engine.")


if __name__ == "__main__":
    main()
