# DarkSight: Enhancing Robotic Perception in Low-Light Environments

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![RealSense](https://img.shields.io/badge/Intel-RealSense-0071C5?style=flat-square)](https://www.intelrealsense.com/)

**DarkSight** is a high-performance, perception-aware low-light enhancement framework I designed for autonomous systems. While I originally based my work on the Zero-3DCE architecture, I have evolved DarkSight into a comprehensive pipeline optimized for real-time robotic vision, temporal stability, and downstream object detection in sub-1 lux environments.

---

## 🌓 Why DarkSight?

Traditional low-light enhancement often focuses on aesthetic quality for human viewers. In this project, **I prioritize machine perception**:

* **Temporal Stability:** ConvGRU memory and motion-aware objectives reduce flickering and maintain feature consistency across video streams.
* **Perception-Aware:** DarkSight is designed to improve downstream perception systems such as YOLOv8 rather than optimize solely for visual quality metrics.
* **Efficient Deployment:** A lightweight recurrent architecture (~76k parameters) enables real-time operation on commodity GPUs.
* **Zero-Reference Training:** The framework learns enhancement without requiring paired ground-truth low-light datasets.
* **Robotics-Oriented Design:** Built and validated on live camera streams using Intel RealSense depth sensing and perception pipelines.

---

## ⚡ Real-Time Optimizations

DarkSight v2 includes several deployment-focused optimizations that enable real-time performance:

### 1. Luma-Chroma Decoupling

Frames are converted from RGB to YCbCr color space.

Only the luminance (Y) channel is enhanced while the original chroma channels (Cb, Cr) are preserved and recombined during post-processing.

Benefits:

* Lower computational cost
* Reduced color artifacts
* Improved deployment efficiency

### 2. Adaptive Downsampling

The luminance channel can optionally be processed at reduced resolution (e.g. 720p → 360p) before enhancement.

Benefits:

* Approximately 4× throughput improvement
* Lower memory footprint
* Faster deployment on edge hardware

### 3. Temporal Memory (ConvGRU)

DarkSight maintains a hidden state across frames using a ConvGRU bottleneck.

Benefits:

* Reduced temporal flicker
* Improved feature stability
* Learned temporal denoising

### 4. Edge-Preserving Denoising

A bilateral filter is applied during post-processing to suppress low-light sensor noise while preserving object boundaries.

Benefits:

* Reduced sparkle noise
* Preserved edges and structure
* Cleaner downstream detections

---

## 🏗 Project Structure

```text
.
├── src/                    # DarkSight Core
│   ├── core/               # Model architecture & DS3DConv
│   ├── training/           # Perception & RAFT-warp losses
│   ├── data/               # Video & Recurrent dataloaders
│   ├── deploy/             # RealSense, TensorRT, & Demo scripts
│   └── eval/               # Perception metrics
├── checkpoints/            # Trained model weights (.pth)
├── exports/                # ONNX and TensorRT engines
├── logs/                   # Training & Demo telemetry
├── Output/                 # Enhanced video results
├── Papers/                 # Reference research papers
└── test_outputs/           # Inference and debug visualizations
```

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU (RTX 30/40 series recommended)
*   Intel RealSense SDK (optional, for D435 support)

### Setup
```bash
git clone https://github.com/gautham-ramkumar/DarkSight
cd DarkSight
# Core dependencies
pip install torch torchvision numpy opencv-python ultralytics lpips onnx onnxruntime-gpu
```

### Usage
**Inference & Validation:**
Run metrics on the LOL validation sets. v2 outputs are saved to `test_outputs/inference/`.
```bash
python3 src/inference.py --checkpoint checkpoints/best.pth
```

**TensorRT Benchmarking:**
Evaluate throughput using the TensorRT/ONNX backends. v2 outputs are saved to `test_outputs/trt_infer/`.
```bash
python3 src/deploy/trt_infer.py --backend trt --engine exports/zero3dce.trt
```

**Real-Time Perception Demo (RGB + Depth + YOLOv8):**
```bash
python3 src/deploy/realsense_demo.py --detect
```

---

## 📊 Performance Benchmark

### Model Size

| Configuration | Parameters |
|---------------|------------|
| Batch Mode | 18,934 (~19k) |
| Recurrent Mode | 76,167 (~76k) |

### Enhancement Performance

| Version | Mode | PSNR (RGB) | ORB Stability |
|----------|----------|----------|----------|
| v1 | Flat | 16.62 dB | 0.98 |
| v2 | Batch | 17.33 dB | 0.99 |
| v2.1 | Recurrent | 17.25 dB | 0.99 |

*Metrics based on `logs/training_log.csv`.*

### Deployment Performance (RTX 4070 Laptop GPU)

| Metric | Value |
|----------|----------|
| Enhancement Core Latency | ~13.9 ms |
| Optimized Enhancement Throughput | 60+ FPS |
| Live RealSense + Detection Pipeline | ~21 FPS |
| ORB Stability Ratio | 0.99 |

**Note:** The 60+ FPS figure refers to the optimized enhancement pipeline using Y-channel processing and adaptive downsampling. The live RealSense demonstration includes sensor acquisition, preprocessing, post-processing, visualization, and YOLOv8 inference, resulting in approximately 21 FPS end-to-end throughput.

---

## 📡 Real-World Perception (D435 Demo)

I validated the system using an Intel RealSense D435 setup in sub-1 lux environments. Telemetry from `logs/demo_telemetry/` shows the following real-world performance:

*   * **Enhancement Core:** ~13.9 ms inference latency using TensorRT FP16.
*   **Live Pipeline:** ~21 FPS end-to-end throughput including camera acquisition, enhancement, detection, and visualization.
*   **Perception Impact:** Stable YOLOv8 detections on enhanced streams where raw low-light inputs frequently produced unreliable or missing detections.
*   **Temporal Stability:** Maintained a 0.99 ORB feature stability ratio across motion and illumination changes.

### 🎥 Enhancement Demo

[![DarkSight Demo](https://img.youtube.com/vi/1y91Fi5i9Fg/0.jpg)](https://youtu.be/1y91Fi5i9Fg)

*Click the image above to watch the full performance demo on YouTube.*

---

## 🛠 Technical Implementation

*   **Model Architecture:** [src/core/model.py](./src/core/model.py) — Implementation of the DS3DConv Encoder-Decoder and ConvGRU recurrent bottleneck.
*   **Training & Losses:** [src/training/losses.py](./src/training/losses.py) — Perception-aware and structural preservation loss functions.
*   **Temporal Stability (RAFT):** [src/training/raft_warp.py](./src/training/raft_warp.py) — Motion-aware optical flow warping for training-time consistency.
*   **Robotic Deployment:** [src/deploy/realsense_demo.py](./src/deploy/realsense_demo.py) — End-to-end pipeline for D435 fusion and YOLOv8 perception.

---

## 🔭 Future Directions

One important observation from deployment testing is that enhancement fundamentally depends on the presence of recoverable visual signal.

While DarkSight performs well in challenging low-light environments, scenes approaching near-zero illumination provide insufficient RGB information for enhancement alone to recover meaningful structure.

This motivates two future research directions:

### Multi-Modal Perception

Integrating thermal sensing alongside RGB inputs to enable perception in no-light environments.

### Representation-Centric Learning

Moving beyond pixel enhancement and toward learning robust scene representations directly from degraded observations.

The long-term goal is not simply to make dark images brighter, but to help autonomous systems understand their environment under extreme illumination conditions.

---

## 📚 References (v2 Implementation)

This project incorporates methodologies and inspirations from the following research found in the [Papers/](./Papers) directory:

1.  **Zero-3DCE:** "A Low-Light Video Enhancement for More Robust Computer Vision Tasks" — [Papers/Zero-3DCE.pdf](./Papers/Zero-3DCE.pdf)
2.  **Zero-DCE:** "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" — [Papers/Zero-DCE.pdf](./Papers/Zero-DCE.pdf)
3.  **RAFT:** "Recurrent All-Pairs Field Transforms for Optical Flow" — [Papers/RAFT.pdf](./Papers/RAFT.pdf)
4.  **FeatEnHancer:** "Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision" — [Papers/FeatEnHancer.pdf](./Papers/FeatEnHancer.pdf)
5.  **Zero-TIG:** "Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement" — [Papers/Zero-TIG.pdf](./Papers/Zero-TIG.pdf)
6.  **Temporally Consistent Enhancement:** "Temporally Consistent Enhancement of Low-Light Videos via Spatial-Temporal Compatible Learning" — [Papers/s11263-024-02084-w.pdf](./Papers/s11263-024-02084-w.pdf)
7.  **LIVENet:** "A Lightweight Video Enhancement Network for Real-time Applications" — [Papers/LIVENet.pdf](./Papers/LIVENet.pdf)
8.  **UPT-Flow:** "Unified Prediction and Transformation for Optical Flow" — [Papers/UPT-Flow.pdf](./Papers/UPT-Flow.pdf)
9.  **STARNet:** "Spatial-Temporal Adaptive Reconstruction Network" — [Papers/STARNet.pdf](./Papers/STARNet.pdf)
10. **BGFlow:** "Background-Guided Optical Flow" — [Papers/BGFlow.pdf](./Papers/BGFlow.pdf)
11. **Spatio-Temporal Reconstruction:** "Digital Signal Processing in Spatio-Temporal Reconstruction" — [Papers/Spatio_temporal_reconstruction.pdf](./Papers/Spatio_temporal_reconstruction.pdf)


---
