# DarkSight: Enhancing Robotic Perception in Low-Light Environments

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![RealSense](https://img.shields.io/badge/Intel-RealSense-0071C5?style=flat-square)](https://www.intelrealsense.com/)

**DarkSight** is a perception-aware low-light video enhancement framework
designed for autonomous systems. Built on the Zero-3DCE architecture,
DarkSight has been evolved into a comprehensive pipeline optimized for
real-time robotic vision, temporal stability, and downstream object
detection in sub-1 lux environments.

---

## 🌓 Why DarkSight?

Traditional low-light enhancement focuses on aesthetic quality for human
viewers. DarkSight prioritizes **machine perception**:

- **Temporal Stability:** ConvGRU memory and motion-aware objectives
  reduce flickering and maintain feature consistency across video streams.
- **Perception-Aware:** Designed to improve downstream perception systems
  such as YOLOv8 rather than optimize solely for visual quality metrics.
- **Efficient Deployment:** A lightweight recurrent architecture (~76k
  parameters) enables real-time operation on commodity GPUs.
- **Zero-Reference Training:** Learns enhancement without requiring paired
  ground-truth low-light datasets.
- **Robotics-Oriented Design:** Built and validated on live Intel RealSense
  D435 streams with integrated YOLOv8 detection.

---

## ⚡ Real-Time Optimizations

### 1. Luma-Chroma Decoupling

Frames are converted from RGB to YCbCr. Only the luminance (Y) channel
is enhanced while chroma channels (Cb, Cr) are preserved and recombined
during post-processing.

Benefits: lower compute cost, reduced color artifacts, improved deployment
efficiency.

### 2. Adaptive Frame Skipping

DarkSight uses a luma-threshold gate — enhancement only runs on frames
below a brightness threshold. Bright frames pass through unmodified.

Benefits: reduces average system load, enables real-time throughput on
edge hardware. Note: reported mean latency figures include skipped frames.
Per-inference latency on dark frames is reported separately below.

### 3. Adaptive Downsampling

The luminance channel is processed at 360p before enhancement and
upsampled back to output resolution.

Benefits: ~4x throughput improvement, lower memory footprint, faster
edge deployment.

### 4. Temporal Memory (ConvGRU)

DarkSight maintains a hidden state across frames using a ConvGRU
bottleneck.

Benefits: reduced temporal flicker, improved feature stability, learned
temporal denoising.

### 5. Edge-Preserving Denoising

A bilateral filter is applied during post-processing to suppress
low-light sensor noise while preserving object boundaries.

Benefits: reduced sparkle noise, preserved edges, cleaner downstream
detections.

---

## 🏗 Project Structure

```text
.
├── src/
│   ├── core/          # Model architecture & DS3DConv
│   ├── training/      # Perception & RAFT-warp losses
│   ├── data/          # Video & recurrent dataloaders
│   ├── deploy/        # RealSense, TensorRT & demo scripts
│   └── eval/          # Perception metrics
├── checkpoints/       # Trained model weights (.pth)
├── exports/           # ONNX and TensorRT engines
├── logs/              # Training & demo telemetry
├── Output/            # Enhanced video results
├── Papers/            # Reference research papers
└── test_outputs/      # Inference and debug visualizations
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 30/40 series recommended)
- Intel RealSense SDK (optional, for D435 support)

### Setup
```bash
git clone https://github.com/gautham-ramkumar/DarkSight
cd DarkSight
pip install torch torchvision numpy opencv-python ultralytics \
            lpips onnx onnxruntime-gpu
```

### Usage

**Inference & Validation:**
```bash
python3 src/inference.py --checkpoint checkpoints/best.pth
```

**TensorRT Benchmarking:**
```bash
python3 src/deploy/trt_infer.py --backend trt \
        --engine exports/zero3dce.trt
```

**Real-Time Perception Demo (RGB + Depth + YOLOv8):**
```bash
python3 src/deploy/realsense_demo.py --detect
```

---

## 📊 Performance

### Model Size

| Configuration  | Parameters      |
|----------------|-----------------|
| Batch Mode     | 18,934 (~19k)   |
| Recurrent Mode | 76,167 (~76k)   |

### Enhancement Quality (LOLv1 / LOLv2 Benchmark)

| Version | Mode      | PSNR (RGB) | ORB Stability |
|---------|-----------|------------|---------------|
| v1      | Flat      | 16.62 dB   | 0.98          |
| v2      | Batch     | 17.33 dB   | 0.99          |
| v2.1    | Recurrent | 17.25 dB   | 0.99          |

*Evaluated on LOLv1 and LOLv2 paired benchmarks.
Training is zero-reference — no paired ground truth used during training.*

### Deployment Performance (RTX 4070 Laptop GPU, PyTorch Backend)

| Metric | Value |
|--------|-------|
| Per-inference latency (dark frames, 360p) | ~57 ms |
| Per-inference FPS (dark frames) | ~17 FPS |
| End-to-end pipeline (RealSense + YOLOv8) | ~21 FPS |
| ORB Stability Ratio | 0.99 |

**Notes on latency reporting:**
- All figures measured with PyTorch backend on RTX 4070 Laptop GPU at
  360p input resolution.
- Adaptive frame skipping means bright frames incur near-zero enhancement
  cost. Mean latency across all frames (including skipped) is ~13.9ms —
  this figure is not representative of per-inference cost on dark frames.
- TensorRT FP16 engine has been exported to `exports/zero3dce.trt`.
  Production deployment targeting sub-20ms per-inference latency pending
  direct hardware benchmark via `trt_infer.py`.

---

## 📡 Real-World Validation (D435 Demo)

Validated on Intel RealSense D435 in sub-1 lux environments:

- **Per-inference latency:** ~57ms on dark frames at 360p (PyTorch backend)
- **End-to-end throughput:** ~21 FPS including sensor acquisition,
  enhancement, YOLOv8 detection, and visualization
- **Perception impact:** Stable YOLOv8 detections on enhanced streams
  where raw low-light inputs produced unreliable or missing detections
- **Temporal stability:** 0.99 ORB feature stability ratio across motion
  and illumination changes

### 🎥 Enhancement Demo

[![DarkSight Demo](https://img.youtube.com/vi/1y91Fi5i9Fg/0.jpg)](https://youtu.be/1y91Fi5i9Fg)

*Click to watch the full performance demo on YouTube.*

---

## 🛠 Technical Implementation

- **Model Architecture:** [src/core/model.py](./src/core/model.py) —
  DS3DConv Encoder-Decoder with ConvGRU recurrent bottleneck
- **Training & Losses:** [src/training/losses.py](./src/training/losses.py) —
  Perception-aware and structural preservation loss functions
- **Temporal Stability:** [src/training/raft_warp.py](./src/training/raft_warp.py) —
  Motion-aware optical flow warping for training-time consistency
- **Robotic Deployment:** [src/deploy/realsense_demo.py](./src/deploy/realsense_demo.py) —
  End-to-end D435 pipeline with YOLOv8 perception

---

## 🔭 Future Directions

Enhancement fundamentally depends on the presence of recoverable visual
signal. Scenes approaching near-zero illumination provide insufficient
RGB information for enhancement alone.

This motivates two research directions:

### Multi-Modal Perception
Integrating thermal sensing alongside RGB to enable perception in
no-light environments.

### Representation-Centric Learning
Moving beyond pixel enhancement toward learning robust scene
representations directly from degraded observations. The long-term goal
is not simply to make dark images brighter, but to help autonomous
systems understand their environment under extreme illumination conditions.

---

## 📚 References

1. **Zero-3DCE** — "A Low-Light Video Enhancement for More Robust
   Computer Vision Tasks"
2. **Zero-DCE** — "Zero-Reference Deep Curve Estimation for Low-Light
   Image Enhancement"
3. **RAFT** — "Recurrent All-Pairs Field Transforms for Optical Flow"
4. **FeatEnHancer** — "Enhancing Hierarchical Features for Object
   Detection Under Low-Light Vision"
5. **Zero-TIG** — "Temporal Consistency-Aware Zero-Shot
   Illumination-Guided Low-light Video Enhancement"
6. **Temporally Consistent Enhancement** — "Temporally Consistent
   Enhancement of Low-Light Videos via Spatial-Temporal Compatible
   Learning"
7. **LIVENet** — "A Lightweight Video Enhancement Network for
   Real-time Applications"
8. **UPT-Flow** — "Unified Prediction and Transformation for
   Optical Flow"
9. **STARNet** — "Spatial-Temporal Adaptive Reconstruction Network"
10. **BGFlow** — "Background-Guided Optical Flow"
11. **Spatio-Temporal Reconstruction** — "Digital Signal Processing
    in Spatio-Temporal Reconstruction"
