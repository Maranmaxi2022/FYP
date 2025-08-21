"""
smp_data.py — Dataset

This dataset reads a single CSV (e.g., `basic_view_pn.csv`) and returns, per sample:
  - "img": 224×224 RGB tensor (cover thumbnail <video_id>.jpg, zeros if missing)
  - "text": [user_id, category, title, keyword, description]  (all strings)
  - "meta": 6-d float tensor = 5 numerics (cols 10..14) + EP (Day-1 popularity if enabled, else 0)
  - "cat":  [category, title]  (strings; title is used for language detection in the model)
  - "id":   video_id (string)
  - "label": sequence tensor of length `seq_len` (popularity scores per day)

Expected CSV layout (0-indexed):
  0:  video_id (str)
  1..4: (unused here)
  5:  user_id (str)
  6:  category (str)   # one of the 15 YouTube categories (English)
  7:  title (str)
  8:  keywords/hashtags (str)
  9:  description (str)
  10..14: five numeric features (float, e.g., duration, followers, posts, title_len, tag_count) — already scaled
  15..: popularity score sequence (Day 1..30), not raw views

Notes
-----
- EP is **not** a separate column; EP = **Day-1** popularity (same unit as labels).
- If `p_i == 1` (use EP), labels start from **Day-2** to avoid leakage.
- `seq_len` is set by the caller:
    • With EP:  use `seq_len = 29` (predict Days 2..30)
    • Without EP: `seq_len = 30` (predict Days 1..30)
- PyTorch's default collate transposes lists, so batches of `text` become 5 lists (one per field),
  which is exactly what `smp_model._bert_features` expects.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

# Allow partially-downloaded images without crashing PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class youtube_data_lstm(Dataset):
    """
    Torch Dataset for SMTPD.

    Parameters
    ----------
    csv_file : str
        Path to the CSV described above.
    cover_dir : str
        Directory containing cover thumbnails named `<video_id>.jpg`.
    gt_path : str
        Unused placeholder (kept for CLI compatibility with older scripts).

    Attributes set by caller
    ------------------------
    seq_len : int
        Prediction horizon (29 or 30). Default: 29.
    p_i : int
        0 = do not use EP; 1 = include EP (Day-1) as the 6th numeric input. Default: 0.

    Modality toggles (rarely changed)
    ---------------------------------
    visual_content, textual_content, numerical_content, categorical_content : bool
    """

    def __init__(self, csv_file: str, cover_dir: str, gt_path: str) -> None:
        self.data = pd.read_csv(csv_file)
        self.cover_dir = cover_dir
        self._cover_suffix = ".jpg"

        # Coerce numeric columns used by the model to float (bad tokens -> NaN, then cleaned later)
        # Columns 10..14 (5 numerics) + labels up to col 44.
        num_cols = list(self.data.columns[10:45])
        self.data[num_cols] = self.data[num_cols].apply(pd.to_numeric, errors="coerce")

        # Basic vision preprocessing (ResNet-101 expects 224×224)
        self.cover_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        # Modality flags
        self.visual_content = True
        self.textual_content = True
        self.numerical_content = True
        self.categorical_content = True

        # EP usage: 0 = don’t use (fill 0), 1 = use Day-1 as EP
        self.p_i = 0

        # Prediction horizon (caller will set to 29 or 30)
        self.seq_len = 29

        # Column index where Day-1 label starts (0-based)
        self._label_start_col = 15

    # ----------------------------- helpers -----------------------------

    def _cover_path(self, video_id: str) -> str:
        return os.path.join(self.cover_dir, f"{video_id}{self._cover_suffix}")

    def _load_cover(self, video_id: str) -> Tensor:
        """Load a 224×224 RGB thumbnail. If missing/broken, return zeros."""
        try:
            img = Image.open(self._cover_path(video_id)).convert("RGB")
            return self.cover_transform(img)
        except Exception:
            return torch.zeros(3, 224, 224)

    # ----------------------------- Dataset API -------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        vid = str(row.iloc[0])

        # ---- Visual (cover) ----
        if self.visual_content:
            img = self._load_cover(vid)
        else:
            img = torch.zeros(3, 224, 224)

        # ---- Text (5 fields) ----
        if self.textual_content:
            user_id = str(row.iloc[5])
            category = str(row.iloc[6])
            title = str(row.iloc[7])
            keyword = str(row.iloc[8])
            description = str(row.iloc[9])
        else:
            user_id = category = title = keyword = description = "0"
        # NOTE: default_collate will transpose this to 5 lists across the batch.
        text = [user_id, category, title, keyword, description]

        # ---- Categorical (category + title for language detection) ----
        if self.categorical_content:
            cat_category = str(row.iloc[6])
            cat_title = str(row.iloc[7])
        else:
            cat_category = cat_title = "0"
        # NOTE: default_collate transposes this to [list_of_categories, list_of_titles]
        cat = [cat_category, cat_title]

        # ---- Numeric features (5) with NaN/Inf guards ----
        if self.numerical_content:
            meta_arr = row.iloc[10:15].to_numpy(dtype=np.float32)
            meta_arr = np.nan_to_num(meta_arr, nan=0.0, posinf=0.0, neginf=0.0)
            meta = torch.from_numpy(meta_arr)
        else:
            meta = torch.zeros(5, dtype=torch.float32)

        # ---- EP (Day-1) as optional 6th numeric ----
        L0 = self._label_start_col  # column of Day-1 label
        if self.p_i == 1:
            ep_val = row.iloc[L0]
            ep_val = 0.0 if pd.isna(ep_val) else float(ep_val)
        else:
            ep_val = 0.0
        ep = torch.tensor([ep_val], dtype=torch.float32)
        meta = torch.cat([meta, ep])  # -> 6 dims

        # ---- Labels sequence (window depends on EP usage) ----
        # With EP: predict Days 2.. (avoid leakage) → start at L0+1
        # Without EP: predict Days 1..            → start at L0
        label_start = L0 + (1 if self.p_i == 1 else 0)
        label_end = label_start + self.seq_len
        lab_arr = row.iloc[label_start:label_end].to_numpy(dtype=np.float32)
        lab_arr = np.nan_to_num(lab_arr, nan=0.0, posinf=0.0, neginf=0.0)
        label = torch.from_numpy(lab_arr)

        sample: Dict[str, Any] = {
            "img": img,          # Tensor [3,224,224]
            "text": text,        # list[str] length=5  (collate -> 5 lists across batch)
            "meta": meta,        # Tensor [6] (5 numerics + EP or 0)
            "cat": cat,          # list[str] length=2  (category, title)
            "id": vid,           # str
            "label": label,      # Tensor [T]
        }
        return sample
