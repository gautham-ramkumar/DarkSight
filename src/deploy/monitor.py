import time
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

class DemoMonitor:
    """
    Real-time performance and quality monitor for Zero-3DCE deployment.
    Tracks:
        - Latency & Jitter (Inference stability)
        - Enhancement Gain (Brightness delta)
        - Perception Utility (Detection counts and confidence)
        - Hardware Stats (via optional psutil)
    """
    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"demo_log_{run_name}_{datetime.now().strftime('%H%M%S')}.csv"
        
        self.history = []
        self.columns = [
            "timestamp", "latency_ms", "fps", "in_luma", "out_luma", 
            "luma_gain", "det_count", "avg_conf", "reset_event"
        ]
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

        # Stats for EMA (Exponential Moving Average)
        self.ema_alpha = 0.1
        self.stats = {k: 0.0 for k in self.columns if k != "timestamp"}
        self.frame_count = 0

    def update(self, latency_ms, in_luma, out_luma, det_count=0, avg_conf=0.0, reset=False):
        self.frame_count += 1
        fps = 1000.0 / max(latency_ms, 1e-6)
        gain = out_luma - in_luma
        
        data = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "fps": fps,
            "in_luma": in_luma,
            "out_luma": out_luma,
            "luma_gain": gain,
            "det_count": det_count,
            "avg_conf": avg_conf,
            "reset_event": 1 if reset else 0
        }
        
        # Update EMA for display
        for k in self.stats:
            if k in data:
                self.stats[k] = (1 - self.ema_alpha) * self.stats[k] + self.ema_alpha * data[k]

        # Periodically write to disk (every 30 frames to avoid I/O lag)
        self.history.append(data)
        if len(self.history) >= 30:
            self._flush()

    def _flush(self):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerows(self.history)
        self.history = []

    def get_display_str(self):
        """Returns a formatted string for the CV2 overlay or terminal."""
        return (f"LAT: {self.stats['latency_ms']:.1f}ms | "
                f"FPS: {self.stats['fps']:.1f} | "
                f"GAIN: +{self.stats['luma_gain']:.2f} | "
                f"DET: {int(self.stats['det_count'])} @ {self.stats['avg_conf']:.2f}")

    def get_summary(self):
        return f"Log saved to: {self.log_file}"
