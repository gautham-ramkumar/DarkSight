#!/usr/bin/env python3
"""
download_videos.py — Download dark/nighttime videos from Pexels and extract
frames for Zero-3DCE temporal training.

Why Pexels instead of YouTube?
-------------------------------
YouTube blocks programmatic downloads (HTTP 429 rate limiting, bot detection).
Pexels provides a free REST API with direct MP4 download links — no yt-dlp,
no cookies, no rate-limit fights. All Pexels videos are CC0 / free to use.

Setup (one-time, ~60 seconds)
------------------------------
1. Go to https://www.pexels.com/api/
2. Sign in with Google / email
3. Copy your API key
4. Pass it with --api-key  OR  set env var PEXELS_API_KEY

Usage
-----
    python3 src/download_videos.py --api-key YOUR_KEY
    python3 src/download_videos.py --api-key YOUR_KEY --n_clips 25 --duration 30 --fps 24
    python3 src/download_videos.py --api-key YOUR_KEY --out Dataset/DarkVideo

Output structure
----------------
    Dataset/DarkVideo/
        clip_001/  frame_0001.png, frame_0002.png, ...   (~720 frames @ 24fps x 30s)
        clip_002/  ...

Requirements
------------
    pip install requests
    # ffmpeg (usually pre-installed):  sudo apt install ffmpeg
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PEXELS_API_BASE = "https://api.pexels.com/videos"

# ---------------------------------------------------------------------------
# Search queries tuned for dark / low-light video content on Pexels.
# Each query targets a different scene type for variety.
# ---------------------------------------------------------------------------
DARK_QUERIES = [
    "night city street",
    "night driving",
    "dark street rain",
    "night road",
    "dark forest night",
    "night market",
    "city lights night",
    "dark indoor",
    "night highway",
    "low light street",
    "rainy night city",
    "dark alley",
    "night sky stars",
    "candlelight dark",
    "neon lights night",
]

RESULTS_PER_QUERY = 5   # Pexels returns up to 80 per page; 5 is plenty


# ---------------------------------------------------------------------------
# Pexels helpers
# ---------------------------------------------------------------------------

def pexels_search(query: str, api_key: str, per_page: int = 5) -> list[dict]:
    """Search Pexels for videos matching query. Returns list of video dicts."""
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
        "size": "medium",          # medium = up to 1080p, avoids 4K files
    }
    try:
        r = requests.get(f"{PEXELS_API_BASE}/search", headers=headers,
                         params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("videos", [])
    except requests.RequestException as e:
        log.warning("Pexels search failed for '%s': %s", query, e)
        return []


def best_download_url(video: dict, max_height: int = 480) -> str | None:
    """Pick the best (≤ max_height) MP4 download link from a Pexels video."""
    files = video.get("video_files", [])
    # Filter to MP4, sort by height descending
    mp4s = [f for f in files if f.get("file_type") == "video/mp4" and
            f.get("height") is not None]
    mp4s.sort(key=lambda f: f["height"], reverse=True)

    # Take the largest file that fits within max_height
    for f in mp4s:
        if f["height"] <= max_height:
            return f["link"]
    # If all are taller, take the smallest available (least bandwidth)
    if mp4s:
        return mp4s[-1]["link"]
    return None


def download_mp4(url: str, dest: Path, timeout: int = 120) -> bool:
    """Stream-download an MP4 file to dest. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    downloaded += len(chunk)
            if total and downloaded < total * 0.95:
                log.warning("  Incomplete download (%d / %d bytes)", downloaded, total)
                return False
            return True
    except requests.RequestException as e:
        log.warning("  Download failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found.  Install with:  sudo apt install ffmpeg")
        sys.exit(1)
    log.info("ffmpeg found ✓")


def video_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames(
    video_path: Path,
    out_dir: Path,
    start_sec: float,
    duration_sec: float,
    fps: int,
    max_height: int = 480,
) -> int:
    """Extract frames as PNG files. Returns number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%04d.png")

    actual_dur = video_duration(video_path)
    start_sec  = min(start_sec, max(0.0, actual_dur - 2.0))
    duration_sec = min(duration_sec, actual_dur - start_sec)

    if duration_sec <= 0:
        log.warning("  Video too short (%.1fs)", actual_dur)
        return 0

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-vf", f"fps={fps},scale=-2:{max_height}",
        "-q:v", "2",
        pattern,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("  ffmpeg error: %s", r.stderr[:200])
        return 0

    return len(list(out_dir.glob("frame_*.png")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download dark/nighttime Pexels videos and extract frames."
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("PEXELS_API_KEY", ""),
        help="Pexels API key (or set env var PEXELS_API_KEY). "
             "Get one free at https://www.pexels.com/api/",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("Dataset/DarkVideo"),
        help="Output directory (default: Dataset/DarkVideo)",
    )
    parser.add_argument(
        "--n_clips", type=int, default=25,
        help="Target number of clips to collect (default: 25)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Seconds of footage to extract per clip (default: 30)",
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Frame extraction rate (default: 24)",
    )
    parser.add_argument(
        "--max_height", type=int, default=480,
        help="Max video resolution height in px (default: 480)",
    )
    parser.add_argument(
        "--start_offset", type=float, default=2.0,
        help="Skip this many seconds at the start of each video (default: 2)",
    )
    args = parser.parse_args()

    # --- Validate API key ---
    if not args.api_key:
        log.error("No Pexels API key provided.")
        log.error("  1. Go to https://www.pexels.com/api/  and sign up (free, ~60s)")
        log.error("  2. Re-run:  python3 src/download_videos.py --api-key YOUR_KEY")
        sys.exit(1)

    check_ffmpeg()

    # Quick API key validation
    test = pexels_search("night city", args.api_key, per_page=1)
    if not test:
        log.error("Pexels API key invalid or network error. Check your key.")
        sys.exit(1)
    log.info("Pexels API key valid ✓")
    log.info("")

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Resume support
    existing = sorted(args.out.glob("clip_*/frame_0001.png"))
    clips_done = len(existing)
    clip_index = clips_done + 1
    if clips_done:
        log.info("Resuming: %d clips already done, continuing from clip_%03d",
                 clips_done, clip_index)

    seen_video_ids: set[int] = set()   # avoid duplicate downloads

    log.info("Target: %d clips  |  %.0fs each  |  %d fps  |  ≤%dp",
             args.n_clips, args.duration, args.fps, args.max_height)
    log.info("")

    for query in DARK_QUERIES:
        if clips_done >= args.n_clips:
            break

        log.info("── Query: '%s' ──", query)
        videos = pexels_search(query, args.api_key, per_page=RESULTS_PER_QUERY)
        if not videos:
            log.warning("  No results, skipping.")
            continue

        for video in videos:
            if clips_done >= args.n_clips:
                break

            vid_id  = video.get("id")
            dur     = video.get("duration", 0)
            author  = video.get("user", {}).get("name", "unknown")

            if vid_id in seen_video_ids:
                continue
            seen_video_ids.add(vid_id)

            if dur < args.duration + args.start_offset:
                log.info("  Skip %s by %s (too short: %ds)", vid_id, author, dur)
                continue

            url = best_download_url(video, args.max_height)
            if not url:
                log.warning("  No suitable MP4 for video %s", vid_id)
                continue

            tmp_file = tmp_dir / f"{vid_id}.mp4"
            log.info("  [%d/%d] id=%s by %s (%.0fs) — downloading...",
                     clips_done + 1, args.n_clips, vid_id, author, dur)

            if not download_mp4(url, tmp_file):
                tmp_file.unlink(missing_ok=True)
                continue

            clip_dir = args.out / f"clip_{clip_index:03d}"
            n_frames = extract_frames(
                tmp_file, clip_dir,
                start_sec    = args.start_offset,
                duration_sec = args.duration,
                fps          = args.fps,
                max_height   = args.max_height,
            )
            tmp_file.unlink(missing_ok=True)

            min_frames = max(args.fps * 2, int(args.duration * args.fps * 0.8))
            if n_frames >= min_frames:
                log.info("  ✓ %d frames → %s", n_frames, clip_dir.name)
                clips_done += 1
                clip_index += 1
            else:
                log.warning("  ✗ Only %d frames (need ≥%d), discarding", n_frames, min_frames)
                shutil.rmtree(clip_dir, ignore_errors=True)

        # Pexels free tier: 200 req/hour — add small pause between queries
        time.sleep(1.0)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Summary ---
    all_clips    = sorted(args.out.glob("clip_*/"))
    total_frames = sum(len(list(c.glob("frame_*.png"))) for c in all_clips)
    total_bytes  = sum(f.stat().st_size for f in args.out.rglob("*.png"))

    log.info("")
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Clips collected : %d / %d", clips_done, args.n_clips)
    log.info("  Total frames    : %d  (~%d D=2 consecutive pairs)",
             total_frames, max(0, total_frames - len(all_clips)))
    log.info("  Disk usage      : %.2f GB", total_bytes / 1e9)
    log.info("")

    if clips_done < args.n_clips:
        log.warning("Fewer clips than requested (%d/%d).", clips_done, args.n_clips)
        log.warning("Re-run to resume — already-downloaded clips are skipped.")
    else:
        log.info("All done! Next step:")
        log.info("  python3 src/train.py")
        log.info("  (RealVideoClipDataset will auto-discover clips in %s)", args.out)


if __name__ == "__main__":
    main()
