import cv2
import numpy as np
import torch
import time
import argparse
from pathlib import Path
import sys

# Import RealSense SDK
try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found. Install it with: pip install pyrealsense2")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy.camera_demo import PyTorchBackend, ONNXBackend, preprocess, postprocess, mean_brightness, put_text, illum_to_panel
from deploy.monitor import DemoMonitor

def run_realsense_demo(backend, width=1280, height=720, threshold=0.35, detect=True):
    # Setup Telemetry
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "demo_telemetry"
    monitor = DemoMonitor(log_dir, run_name="D435_Perception")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    # Start streaming
    print(f"\n[RealSense] Starting D435 pipeline (RGB + Depth)...")
    profile = pipeline.start(config)
    
    # YOLOv8 for real-time detection
    yolo = None
    if detect:
        try:
            from ultralytics import YOLO
            yolo = YOLO('yolov8n.pt') # Use nano for speed
            print("  [ok] YOLOv8n loaded for real-time detection.")
        except ImportError:
            print("  [warn] ultralytics not found - detection disabled.")

    # Colormap for depth
    colorizer = rs.colorizer()

    # Setup Window
    display_w, display_h = 640, int(640 * (height/width))
    win = "Zero3DCE v2.2 [Perception Demo: RGB + Depth + Detect]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w * 3, display_h)

    prev_brightness = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # ── Preprocess ──────────────────────────────────────────────────
            t_cur, pad_h, pad_w, orig_ycrcb = preprocess(frame, backend.device, downsample=True)
            x = t_cur if backend.is_recurrent else torch.cat([t_cur, t_cur], dim=2)

            # ── Decision & Inference ─────────────────────────────────────────
            brightness = mean_brightness(frame)
            is_dark    = brightness < threshold
            
            t0 = time.perf_counter()
            if is_dark:
                enhanced_t = backend.run(x)
            else:
                enhanced_t = t_cur
            
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Postprocess & Detection ──────────────────────────────────────
            det_count = 0
            avg_conf = 0.0

            if is_dark:
                show_right = postprocess(enhanced_t, display_h, display_w, pad_h, pad_w, orig_ycrcb)
                # Run YOLO on enhanced frame
                if yolo:
                    results = yolo(show_right, verbose=False, conf=0.25)
                    show_right = results[0].plot()
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        det_count = len(boxes)
                        avg_conf = float(boxes.conf.mean())
                put_text(show_right, "Enhanced + Detect", (8, 24), (80, 255, 80))
            else:
                show_right = cv2.resize(frame, (display_w, display_h))
                if yolo:
                    results = yolo(show_right, verbose=False, conf=0.25)
                    show_right = results[0].plot()
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        det_count = len(boxes)
                        avg_conf = float(boxes.conf.mean())
                put_text(show_right, "Passthrough + Detect", (8, 24), (80, 220, 80))

            # ── Update Monitor ───────────────────────────────────────────────
            out_brightness = mean_brightness(show_right)
            monitor.update(elapsed_ms, brightness, out_brightness, det_count, avg_conf)

            put_text(show_right, monitor.get_display_str(), (8, display_h-12), (255, 255, 80))

            # ── Panels ───────────────────────────────────────────────────────
            show_left  = cv2.resize(frame, (display_w, display_h))
            show_depth = cv2.resize(depth_image, (display_w, display_h))
            div = np.zeros((display_h, 4, 3), dtype=np.uint8)
            
            canvas = np.hstack([show_left, div, show_depth, div, show_right])
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): backend.reset_state()

    finally:
        print(f"\n[Terminating] {monitor.get_summary()}")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to zero3dce_recurrent.onnx")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--detect", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = ONNXBackend(args.onnx, device)
    
    run_realsense_demo(backend, args.width, args.height, detect=args.detect)
