# DarkSight: Enhancing Robotic Perception in Low-Light Environments

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![RealSense](https://img.shields.io/badge/Intel-RealSense-0071C5?style=flat-square)](https://www.intelrealsense.com/)

**DarkSight** is a high-performance, perception-aware low-light enhancement framework I designed for autonomous systems. While I originally based my work on the Zero-3DCE architecture, I have evolved DarkSight into a comprehensive pipeline optimized for real-time robotic vision, temporal stability, and downstream object detection in sub-1 lux environments.

---

## 🌓 Why DarkSight?

Traditional low-light enhancement often focuses on aesthetic quality for human viewers. In this project, **I prioritize machine perception**:
*   **Temporal Stability:** I eliminate flickering in video streams using ConvGRU memory and motion-aware RAFT losses.
*   **Perception-Aware:** I specifically tuned my model to improve the accuracy of downstream detectors like YOLOv8.
*   **Real-Time Performance:** I achieve 60+ FPS on 720p streams using TensorRT and optimized 3D convolutions.
*   **Zero-Reference:** I trained my system to enhance without needing "perfect" ground-truth labels, making it adaptable to diverse real-world sensors.

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

| Version | Mode | PSNR (RGB) | ORB Stability | FPS (720p) |
|---|---|---|---|---|
| v1 | Flat | 16.62 dB | 0.98 | 45 |
| v2 | Batch | 17.33 dB | 0.99 | 58 |
| **v2.1** | **Recurrent** | **17.25 dB** | **0.99** | **60+** |

*Metrics based on `logs/training_log.csv`. PSNR reported for full RGB; Y-channel PSNR peaks at **18.32 dB**.*

---

## 📡 Real-World Perception (D435 Demo)

I validated the system using an Intel RealSense D435 setup in sub-1 lux environments. Telemetry from `logs/demo_telemetry/` shows the following real-world performance:

*   **Inference Latency:** Average of **13.9ms** for the enhancement core.
*   **End-to-End Perception:** Achieved stable object detection (YOLOv8) on enhanced streams where the raw input showed zero detections.
*   **Reliability:** Successfully maintained a **0.99 stability ratio** across sustained motion and lighting transitions.

### 🎥 Enhancement Demo
<video src="Output/Outdoor_env.webm" width="100%" controls></video>

---

## 🛠 Technical Implementation

*   **Model Architecture:** [src/core/model.py](./src/core/model.py) — Implementation of the DS3DConv Encoder-Decoder and ConvGRU recurrent bottleneck.
*   **Training & Losses:** [src/training/losses.py](./src/training/losses.py) — Perception-aware and structural preservation loss functions.
*   **Temporal Stability (RAFT):** [src/training/raft_warp.py](./src/training/raft_warp.py) — Motion-aware optical flow warping for training-time consistency.
*   **Robotic Deployment:** [src/deploy/realsense_demo.py](./src/deploy/realsense_demo.py) — End-to-end pipeline for D435 fusion and YOLOv8 perception.

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
