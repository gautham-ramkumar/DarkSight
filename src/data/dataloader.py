import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE  = 256
BATCH_SIZE  = 16
NUM_WORKERS = 4

# ---------------------------------------------------------------------------
# Dataset root
# ---------------------------------------------------------------------------
DATASET_ROOT    = Path(__file__).resolve().parent.parent.parent / "Dataset"
DARK_VIDEO_ROOT = DATASET_ROOT / "DarkVideo"

# ---------------------------------------------------------------------------
# Training: image datasets (D=1 clips, zero-reference — low images only)
# ---------------------------------------------------------------------------
TRAIN_IMAGE_CONFIGS = {
    "LOLv1": {
        "low":  str(DATASET_ROOT / "LOLv1/lol_dataset/our485/low/*"),
        "high": str(DATASET_ROOT / "LOLv1/lol_dataset/our485/high/*"),
    },
    "LOLv2": {
        "low":  str(DATASET_ROOT / "LOLv2/LOL-v2/Real_captured/Train/Low/*"),
        "high": str(DATASET_ROOT / "LOLv2/LOL-v2/Real_captured/Train/Normal/*"),
    },
}

# ---------------------------------------------------------------------------
# Validation: paired image datasets (D=1 clips)
# ---------------------------------------------------------------------------
VAL_CONFIGS = {
    "LOLv1": {
        "low":  str(DATASET_ROOT / "LOLv1/lol_dataset/eval15/low/*"),
        "high": str(DATASET_ROOT / "LOLv1/lol_dataset/eval15/high/*"),
    },
    "LOLv2": {
        "low":  str(DATASET_ROOT / "LOLv2/LOL-v2/Real_captured/Test/Low/*"),
        "high": str(DATASET_ROOT / "LOLv2/LOL-v2/Real_captured/Test/Normal/*"),
    },
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Paired transforms — MUST apply identical spatial ops to both images/frames
# ---------------------------------------------------------------------------
def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image → float32 tensor (C, H, W) in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def train_transform(img1: Image.Image, img2: Image.Image, crop_size: int = IMAGE_SIZE):
    """
    Joint augmentation for a pair of images (or frames).
    All spatial operations use the same random parameters so the pair
    stays perfectly aligned after augmentation.
    """
    w, h = img1.size
    scale = crop_size / min(w, h)
    new_w = max(crop_size, int(w * scale))
    new_h = max(crop_size, int(h * scale))
    img1 = img1.resize((new_w, new_h), Image.BILINEAR)
    img2 = img2.resize((new_w, new_h), Image.BILINEAR)

    # max(0, ...) guards against int() truncation of the scale product
    x = random.randint(0, max(0, new_w - crop_size))
    y = random.randint(0, max(0, new_h - crop_size))
    img1 = img1.crop((x, y, x + crop_size, y + crop_size))
    img2 = img2.crop((x, y, x + crop_size, y + crop_size))

    if random.random() < 0.5:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

    rotation_map = {1: Image.Transpose.ROTATE_90,
                    2: Image.Transpose.ROTATE_180,
                    3: Image.Transpose.ROTATE_270}
    k = random.randint(0, 3)
    if k in rotation_map:
        img1 = img1.transpose(rotation_map[k])
        img2 = img2.transpose(rotation_map[k])

    return _to_tensor(img1), _to_tensor(img2)


def val_transform(img1: Image.Image, img2: Image.Image, crop_size: int = IMAGE_SIZE):
    """Deterministic centre-crop for validation."""
    w, h = img1.size
    scale = crop_size / min(w, h)
    new_w = max(crop_size, int(w * scale))
    new_h = max(crop_size, int(h * scale))
    img1 = img1.resize((new_w, new_h), Image.BILINEAR)
    img2 = img2.resize((new_w, new_h), Image.BILINEAR)

    x = max(0, (new_w - crop_size) // 2)
    y = max(0, (new_h - crop_size) // 2)
    img1 = img1.crop((x, y, x + crop_size, y + crop_size))
    img2 = img2.crop((x, y, x + crop_size, y + crop_size))

    return _to_tensor(img1), _to_tensor(img2)


# ---------------------------------------------------------------------------
# Training Dataset A: Real Video Clips (D=2 consecutive pairs)
# ---------------------------------------------------------------------------
class RealVideoClipDataset(Dataset):
    """
    Scans Dataset/DarkVideo/clip_*/ and extracts pairs of consecutive frames.
    Yields (C, 2, H, W) tensors — true temporal pairs, not duplicated images.
    """
    def __init__(self, root_dir: Path, transform):
        self.transform = transform
        self.pairs = []

        if not root_dir.exists():
            raise FileNotFoundError(f"DarkVideo root not found: {root_dir}")

        skipped_clips = 0
        for clip in sorted(root_dir.glob("clip_*")):
            frames = sorted(clip.glob("frame_*.png"))
            if len(frames) < 2:
                skipped_clips += 1
                continue
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        if skipped_clips:
            print(f"[warn] Skipped {skipped_clips} clips with < 2 frames in {root_dir}")
        if not self.pairs:
            raise RuntimeError(f"No video frames found in {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            f1_path, f2_path = self.pairs[idx]
            img1 = Image.open(f1_path).convert("RGB")
            img2 = Image.open(f2_path).convert("RGB")
            t1, t2 = self.transform(img1, img2)
            clip = torch.stack([t1, t2], dim=1)   # (C, 2, H, W)
            return clip, clip
        except Exception as e:
            import warnings
            warnings.warn(f"[RealVideoClipDataset] Corrupt pair at idx {idx}: {e} — substituting idx 0")
            if idx != 0:
                return self[0]
            t = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            clip = torch.stack([t, t], dim=1)
            return clip, clip


# ---------------------------------------------------------------------------
# Training Dataset A2: Long Video Sequences for ConvGRU / Recurrent Training
# ---------------------------------------------------------------------------
class RecurrentVideoClipDataset(Dataset):
    """
    Loads sequences of `seq_len` consecutive frames from DarkVideo clips.
    Yields (C, seq_len, H, W) tensors for ConvGRU recurrent training.

    Uses a sliding window (stride=1) over each clip to maximise data.
    Applies spatially-consistent augmentation across all frames in the sequence
    so they remain perfectly aligned after random crop / flip / rotate.
    """
    def __init__(self, root_dir: Path, seq_len: int = 8, stride: int = None,
                 image_size: int = IMAGE_SIZE):
        self.seq_len    = seq_len
        self.stride     = stride if stride is not None else max(1, seq_len // 2)
        self.image_size = image_size
        self.sequences  = []   # list of lists of Path objects

        if not root_dir.exists():
            raise FileNotFoundError(f"DarkVideo root not found: {root_dir}")

        skipped_clips = 0
        for clip in sorted(root_dir.glob("clip_*")):
            frames = sorted(clip.glob("frame_*.png"))
            if len(frames) < seq_len:
                skipped_clips += 1
                continue
            for i in range(0, len(frames) - seq_len + 1, self.stride):
                self.sequences.append(frames[i : i + seq_len])

        if skipped_clips:
            print(f"[warn] Skipped {skipped_clips} clips with < {seq_len} frames in {root_dir}")
        if not self.sequences:
            raise RuntimeError(
                f"No sequences of length {seq_len} found in {root_dir}. "
                f"Try a shorter seq_len."
            )

    def __len__(self):
        return len(self.sequences)

    def _consistent_transform(self, imgs: list) -> list[torch.Tensor]:
        """Apply one set of random spatial transforms to every frame."""
        s   = self.image_size
        w0, h0 = imgs[0].size
        scale  = s / min(w0, h0)
        new_w  = max(s, int(w0 * scale))
        new_h  = max(s, int(h0 * scale))

        x_off    = random.randint(0, max(0, new_w - s))
        y_off    = random.randint(0, max(0, new_h - s))
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        rot_k    = random.randint(0, 3)
        rot_map  = {1: Image.Transpose.ROTATE_90,
                    2: Image.Transpose.ROTATE_180,
                    3: Image.Transpose.ROTATE_270}

        out = []
        for img in imgs:
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = img.crop((x_off, y_off, x_off + s, y_off + s))
            if do_hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if do_vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if rot_k in rot_map:
                img = img.transpose(rot_map[rot_k])
            out.append(_to_tensor(img))
        return out

    def __getitem__(self, idx):
        try:
            imgs = [Image.open(p).convert("RGB") for p in self.sequences[idx]]
            tensors = self._consistent_transform(imgs)
            clip = torch.stack(tensors, dim=1)   # (C, seq_len, H, W)
            return clip, clip
        except Exception as e:
            import warnings
            warnings.warn(f"[RecurrentVideoClipDataset] Error at idx {idx}: {e} — substituting idx 0")
            if idx != 0:
                return self[0]
            t = torch.zeros(3, self.image_size, self.image_size)
            clip = t.unsqueeze(1).expand(-1, self.seq_len, -1, -1).contiguous()
            return clip, clip


# ---------------------------------------------------------------------------
# Training Dataset B: Low-Light Images as D=1 Clips (zero-reference)
# ---------------------------------------------------------------------------
class LowLightSingleFrameDataset(Dataset):
    """
    Loads unpaired low-light images as D=1 clips → (C, 1, H, W).
    Zero-reference training: no ground-truth required.
    """
    def __init__(self, low_glob: str, transform):
        self.transform = transform
        self.low_paths = sorted(
            p for p in glob(low_glob)
            if Path(p).suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.low_paths:
            raise RuntimeError(f"No images found at: {low_glob}")

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.low_paths[idx]).convert("RGB")
            t, _ = self.transform(img, img)
            clip = t.unsqueeze(1)   # (C, H, W) → (C, 1, H, W)
            return clip, clip
        except Exception as e:
            import warnings
            warnings.warn(f"[LowLightSingleFrameDataset] Corrupt image at idx {idx}: {e} — substituting idx 0")
            if idx != 0:
                return self[0]
            t = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return t.unsqueeze(1), t.unsqueeze(1)


# ---------------------------------------------------------------------------
# Validation Dataset: Paired Images as D=1 Clips
# ---------------------------------------------------------------------------
class LowLightImageDataset(Dataset):
    """Paired low/high image dataset (2D)."""
    def __init__(self, low_glob: str, high_glob: str, transform):
        self.transform = transform
        self.low_paths  = sorted(p for p in glob(low_glob)  if Path(p).suffix.lower() in _IMAGE_EXTENSIONS)
        self.high_paths = sorted(p for p in glob(high_glob) if Path(p).suffix.lower() in _IMAGE_EXTENSIONS)
        if len(self.low_paths) != len(self.high_paths):
            raise RuntimeError(
                f"Paired dataset mismatch: {len(self.low_paths)} low vs "
                f"{len(self.high_paths)} high images.\n"
                f"  low  glob : {low_glob}\n"
                f"  high glob : {high_glob}"
            )

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low  = Image.open(self.low_paths[idx]).convert("RGB")
        high = Image.open(self.high_paths[idx]).convert("RGB")
        return self.transform(low, high)   # (C, H, W), (C, H, W)


class SingleFrameClipDataset(Dataset):
    """
    Wraps a paired 2D image dataset as D=1 clips.
    (C, H, W) → (C, 1, H, W) — no frame duplication.
    """
    def __init__(self, image_dataset: Dataset):
        self.dataset = image_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        low, high = self.dataset[idx]               # (C, H, W)
        return low.unsqueeze(1), high.unsqueeze(1)  # (C, 1, H, W)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_video_train_dataloader(
    batch_size:  int  = BATCH_SIZE,
    image_size:  int  = IMAGE_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = True,
) -> DataLoader:
    """DarkVideo consecutive frame pairs → D=2 clips."""
    print("Building DarkVideo video training dataset (D=2) …")
    tfm = lambda img1, img2: train_transform(img1, img2, image_size)
    ds  = RealVideoClipDataset(DARK_VIDEO_ROOT, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    print(f"  Video train : {len(ds):,} pairs | {len(loader)} batches/epoch\n")
    return loader


def get_image_train_dataloader(
    batch_size:  int  = BATCH_SIZE,
    image_size:  int  = IMAGE_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = True,
) -> DataLoader:
    """LOLv1 (our485) + LOLv2 (Train) low images only → D=1 clips (zero-reference)."""
    print("Building image training datasets (D=1, zero-reference) …")
    tfm = lambda img1, img2: train_transform(img1, img2, image_size)
    sub_datasets = []
    for name, cfg in TRAIN_IMAGE_CONFIGS.items():
        ds = LowLightSingleFrameDataset(cfg["low"], transform=tfm)
        sub_datasets.append(ds)
        print(f"  [{name}/train]  {len(ds):,} images → D=1 clips")
    concat_ds = ConcatDataset(sub_datasets)
    loader = DataLoader(
        concat_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    print(f"  Image train : {len(concat_ds):,} total | {len(loader)} batches/epoch\n")
    return loader


def get_val_dataloader(
    batch_size:  int  = BATCH_SIZE,
    image_size:  int  = IMAGE_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = True,
) -> DataLoader:
    """LOLv1 eval15 + LOLv2 Test → D=1 clips (no frame duplication)."""
    print("Building validation datasets (D=1) …")
    tfm = lambda low, high: val_transform(low, high, image_size)
    sub_datasets = []
    for name, cfg in VAL_CONFIGS.items():
        base_ds = LowLightImageDataset(cfg["low"], cfg["high"], transform=tfm)
        clip_ds = SingleFrameClipDataset(base_ds)
        sub_datasets.append(clip_ds)
        print(f"  [{name}/val]  {len(base_ds)} images → D=1 clips")
    concat_ds = ConcatDataset(sub_datasets)
    loader = DataLoader(
        concat_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )
    print(f"  Val total   : {len(concat_ds)} clips | {len(loader)} batches/epoch\n")
    return loader


def get_video_metric_loader(
    batch_size:  int  = 4,
    image_size:  int  = IMAGE_SIZE,
    num_workers: int  = NUM_WORKERS,
    max_samples: int  = 200,
    pin_memory:  bool = True,
) -> DataLoader:
    """
    DarkVideo consecutive frame pairs for temporal metric evaluation (no augmentation).

    Uses val_transform (deterministic centre-crop) and limits to the first
    `max_samples` pairs so metric computation stays fast.  Shuffle=False
    ensures consistent results across epochs.
    """
    from torch.utils.data import Subset
    tfm = lambda img1, img2: val_transform(img1, img2, image_size)
    ds  = RealVideoClipDataset(DARK_VIDEO_ROOT, transform=tfm)
    n   = min(max_samples, len(ds))
    sub = Subset(ds, list(range(n)))
    loader = DataLoader(
        sub,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )
    print(f"  Video metric: {n}/{len(ds)} pairs | {len(loader)} batches\n")
    return loader


def get_recurrent_train_dataloader(
    batch_size:  int  = 2,          # reduce vs D=2 — D=8 clips use ~4× more memory
    seq_len:     int  = 8,
    stride:      int  = None,
    image_size:  int  = IMAGE_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = True,
) -> DataLoader:
    """DarkVideo sequences of `seq_len` frames → D=seq_len clips for ConvGRU training."""
    print(f"Building DarkVideo recurrent training dataset (D={seq_len}, stride={stride or seq_len//2}) …")
    ds = RecurrentVideoClipDataset(DARK_VIDEO_ROOT, seq_len=seq_len, stride=stride, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    print(f"  Recurrent train : {len(ds):,} sequences | {len(loader)} batches/epoch\n")
    return loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    video_loader     = get_video_train_dataloader(batch_size=4, num_workers=0)
    image_loader     = get_image_train_dataloader(batch_size=4, num_workers=0)
    val_loader       = get_val_dataloader(batch_size=4,         num_workers=0)
    recurrent_loader = get_recurrent_train_dataloader(batch_size=2, seq_len=8, num_workers=0)

    low_vid, _ = next(iter(video_loader))
    print(f"Video train batch      — shape: {low_vid.shape}, dtype: {low_vid.dtype}")

    low_img, _ = next(iter(image_loader))
    print(f"Image train batch      — shape: {low_img.shape}, dtype: {low_img.dtype}")

    low_val, high_val = next(iter(val_loader))
    print(f"Val batch              — low: {low_val.shape}, high: {high_val.shape}")
    print(f"  D=1 confirmed        : {low_val.shape[2] == 1}")

    low_rec, _ = next(iter(recurrent_loader))
    print(f"Recurrent train batch  — shape: {low_rec.shape}")
    print(f"  D=8 confirmed        : {low_rec.shape[2] == 8}")
