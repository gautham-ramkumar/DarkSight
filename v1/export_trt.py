"""
DarkSight v1: Export to ONNX / TensorRT
Simplified for the baseline 7-layer flat architecture.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure we can import from root and v1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v1.model import FlatZero3DCE

class V1ExportWrapper(nn.Module):
    """Wrapper to return (alpha_maps, enhanced) for ONNX tracing."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--output", type=str, default="v1/model_v1.onnx")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = FlatZero3DCE().to(device)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    wrapper = V1ExportWrapper(model)
    
    # Dummy input (B=1, C=3, D=2, H, W)
    dummy_x = torch.rand(1, 3, 2, args.image_size, args.image_size, device=device)
    
    # Export
    print(f"Exporting DarkSight v1 to {args.output}...")
    torch.onnx.export(
        wrapper,
        dummy_x,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["alpha_maps", "enhanced"],
        dynamic_axes={
            "input": {0: "batch", 3: "height", 4: "width"},
            "alpha_maps": {0: "batch", 3: "height", 4: "width"},
            "enhanced": {0: "batch", 3: "height", 4: "width"},
        }
    )
    print("Done. To convert to TensorRT, use trtexec:")
    print(f"  trtexec --onnx={args.output} --saveEngine=v1/model_v1.trt --fp16")

if __name__ == "__main__":
    main()
