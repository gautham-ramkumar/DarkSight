# DarkSight v1: Legacy Baseline — Technical Deep Dive

Inspired by the original **Zero-DCE** and **Zero-3DCE** (Zero-Reference Deep Curve Estimation), v1 was my first attempt to extend frame-level enhancement to video using 3D convolutions. It served as the foundation for the DarkSight project, establishing the viability of zero-reference learning for robotic perception.

## 🏗 Architecture
The v1 model is a **7-layer flat 3D Convolutional Network**.
- **No Downsampling:** Unlike typical U-Nets, v1 maintains full spatial resolution in every layer. This was designed to prevent the loss of high-frequency textures and edges that often occurs during pooling.
- **Symmetric Skip Connections:** Features from the first three layers are concatenated with the outputs of the later layers. This allows the network to combine low-level structural details with higher-level feature representations before estimating the final enhancement curves.
- **Parameter Count:** Extremely efficient at ~18,934 parameters.
- **Temporal Window:** Fixed at $D=2$ (processing pairs of consecutive frames).

## 💡 Inspiration: From 2D to 3D
The primary inspiration was the **Zero-DCE** paper, which demonstrated that a network could learn to enhance images by optimizing for perceptual properties (exposure, color, contrast) rather than pixel-to-pixel similarity to a ground truth. 

I saw an opportunity in **robotic video streams**: by replacing 2D kernels with 3D kernels, I could force the network to maintain **temporal coherence**. My goal was to ensure that the enhancement applied to Frame A was consistent with Frame B, reducing the "flicker" common in frame-by-frame processing.

## 🚀 Training Process
v1 was trained using a **Pure Zero-Reference** strategy on the LOL and internet scraped datasets:
1.  **Exposure Control:** Local patches were guided toward a target brightness of 0.6.
2.  **Color Constancy:** RGB channels were balanced to ensure no single color dominated the scene.
3.  **Spatial Consistency:** The gradients of the input were enforced on the output to preserve edges.
4.  **Naive Temporal Smoothness:** A Total Variation (TV) loss across the temporal dimension penalized sudden changes in the predicted alpha maps.

## ⚠️ Problems Faced
*   **The "0.5 Bias" Trap:** I discovered that initializing the output bias to 0.5 caused an "initialization artifact." The model started with a $tanh(0.5)$ boost that looked good on day one but didn't actually represent learning. Real progress only happened when I reset bias to 0.0.
*   **Optimization Drift:** When I tried to mix zero-reference losses with L1 supervision (using LOL ground truth), the model suffered. The Zero-Ref loss wanted perceptual "perfection," while L1 wanted pixel-matching. This conflict caused the PSNR to peak early and then steadily decline as the model chose one objective over the other.
*   **Motion vs. Stability:** The naive temporal loss couldn't distinguish between **camera flicker** and **actual motion**. If a car moved across the frame, the v1 loss penalized the model for changing the pixels, leading to ghosting artifacts.

## ✅ Pros & Cons
### Pros
- **Detail Fidelity:** Zero spatial information loss due to the flat architecture.
- **Speed:** Very low latency on high-end GPUs due to small parameter count.
- **No Artifacts:** Avoids the "checkerboard" artifacts sometimes seen in encoder-decoder networks.

### Cons
- **Limited Receptive Field:** Because there is no pooling, the model can only "see" a small neighborhood of pixels. It cannot understand global lighting contexts (e.g., a bright window on the other side of a dark room).
- **VRAM Heavy:** Processing 3D tensors at full resolution is computationally expensive and scales poorly to 4K resolutions.
- **Short Memory:** With only a 2-frame window, the model has no "memory" of what happened a second ago.

## ➡️ The Move to v2
I moved to the **Encoder-Decoder + ConvGRU (v2)** architecture for three reasons:
1.  **Global Context:** Downsampling (pooling) allows the model to see the "big picture" of the scene's lighting.
2.  **Recurrent Memory:** Replacing the flat window with a **ConvGRU** allows the model to carry state across thousands of frames, handling sustained transitions like entering a tunnel.
3.  **Perception-Awareness:** I transitioned the focus from "human-readable" enhancement to "machine-usable" enhancement, integrating YOLOv8 and RAFT optical flow into the training loop.
