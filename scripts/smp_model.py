"""
smp_model.py — SMTPD baseline multi-modal sequence regressor (Fig. 5 in the paper)

Architecture (per sample)
-------------------------
Inputs:
  • Visual (cover image): ResNet-101 backbone → 2048-D → MLP → 128-D
  • Textual (5 fields: user_id, category, title, keywords, description):
      mBERT pooled outputs (each 768-D) → stack → 1×1 conv over fields → 768-D → MLP → 128-D
  • Numerical (5 features + optional EP): MLP → 128-D
  • Categorical (category + language-from-title):
      category embedding (128) & language embedding (128) → MLPs → element-wise product → MLP → 128-D

Fusion:
  Concatenate [visual(128), categorical(128), textual(128), numeric(128)] → 512-D

Temporal regressor:
  • Initial (h0, c0) from fused vector via MLP
  • Unrolled LSTMCell for T=seq_length steps
  • Per-step head: [h_t, c_t] → MLP → score_t (clamped ≥ 0)

Notes
-----
- mBERT and ResNet feature extraction are done under `torch.no_grad()` to save memory.
- Category strings in real CSVs can have minor spacing/punctuation differences; we normalize keys for lookup.
- Language is inferred from the *title* using `langid` (kept as in your original code).

Exports
-------
- youtube_lstm3: nn.Module producing a [B, T] tensor of popularity scores.
"""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

import torch
from torch import nn, Tensor
from torchvision import models
from transformers import BertTokenizer, BertModel, logging as hf_logging
import langid

# -------------------------------------------------------------------
# Device & HF logging
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_logging.set_verbosity_error()


# -------------------------------------------------------------------
# Utilities: BERT loading & text features
# -------------------------------------------------------------------
def _bert(model_or_path: str | None = None) -> Tuple[BertTokenizer, BertModel]:
    """
    Load tokenizer + model for multilingual BERT.

    Environment override:
        Set BERT_PATH to a local/alternate model (e.g., a cached path).
    """
    name = model_or_path or os.getenv("BERT_PATH", "bert-base-multilingual-cased")
    tok = BertTokenizer.from_pretrained(name)
    mdl = BertModel.from_pretrained(name).to(device)
    return tok, mdl


class _TextConv(nn.Module):
    """
    1×1 convolution across the 5 text fields to learn a weighted combination.

    Input:  [B, T=5, 768, 1]   (stacked pooled outputs from mBERT)
    Output: [B, 768]
    """
    def __init__(self, text_num: int = 5) -> None:
        super().__init__()
        self.text_num = text_num
        self.conv = nn.Conv2d(text_num, 1, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=1.0 / text_num, std=0.01)

    def forward(self, stacked_bert: Tensor) -> Tensor:  # [B, 5, 768, 1]
        x = self.conv(stacked_bert)                     # [B, 1, 768, 1]
        x = x.permute(0, 2, 1, 3).squeeze()            # [B, 768]
        return x


def _bert_features(
    texts: Sequence[Sequence[str]],
    tok: BertTokenizer,
    mdl: BertModel,
    max_len: int = 64,
) -> Tensor:
    """
    Batched mBERT pooling for 5 textual fields.

    Args
    ----
    texts : Sequence of 5 sequences, each length B (strings).
            This is how PyTorch's default_collate transposes batches of lists.
    tok   : mBERT tokenizer
    mdl   : mBERT model
    max_len : max tokens per field

    Returns
    -------
    Tensor with shape [B, 5, 768, 1] for downstream 1×1 conv.
    """
    out: List[Tensor] = []
    for field in texts:
        enc = tok(list(field), padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            pooled = mdl(**enc).pooler_output  # [B, 768]
        out.append(pooled)
    return torch.stack(out, dim=1).unsqueeze(3)  # [B, 5, 768, 1]


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class youtube_lstm3(nn.Module):
    """
    SMTPD baseline multi-modal sequence regressor (see module docstring).

    Parameters
    ----------
    seq_length : int
        Number of time steps (days) to predict.
    batch_size : int
        Kept for API compatibility; not directly used in forward.
    """
    def __init__(self, seq_length: int, batch_size: int) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.text_num = 5
        self.meta_num = 6  # 5 numerics + EP (or 0)

        # ---------------- Visual ----------------
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.img_feature = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 2048, 1, 1]
        self.img_mlp = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512,  affine=False), nn.ReLU(),
            nn.Linear(512, 128),
        )

        # ---------------- Text (mBERT) ----------------
        self.tok, self.bert = _bert()
        self.text_conv = _TextConv(text_num=self.text_num)
        self.text_mlp = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(),
            nn.Linear(512, 128),
        )

        # ---------------- Categorical ----------------
        # Normalize keys to be robust to spacing/punctuation variants
        def _norm_key(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        # Canonical YouTube categories used in the dataset (normalized)
        self._cate_vocab_norm = {
            _norm_key("People & Blogs"):          1,
            _norm_key("Gaming"):                  2,
            _norm_key("News & Politics"):         3,
            _norm_key("Entertainment"):           4,
            _norm_key("Music"):                   5,
            _norm_key("Education"):               6,
            _norm_key("Sports"):                  7,
            _norm_key("Howto & Style"):           8,
            _norm_key("Film & Animation"):        9,
            _norm_key("Nonprofits & Activism"):   10,
            _norm_key("Travel"):                  11,
            _norm_key("Comedy"):                  12,
            _norm_key("Science & Technology"):    13,
            _norm_key("Autos & Vehicles"):        14,
            _norm_key("Pets & Animals"):          15,
        }
        # Backward-compatible fallback (index 0)
        self.cate_ooa_idx = 0

        # A small language vocab (extend as needed)
        self.lang_vocab = {"en": 1, "zh": 2, "ko": 3, "ja": 4, "hi": 5, "ru": 6, "OOA": 0}

        self.cate_embed = nn.Embedding(16, 128)  # 0..15
        self.lang_embed = nn.Embedding(7, 128)   # 0..6

        self.cate_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256, affine=False), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.lang_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256, affine=False), nn.ReLU(),
            nn.Linear(256, 128),
        )
        # After element-wise product (⊙) per Eq. (9) in the paper
        self.emb_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256, affine=False), nn.ReLU(),
            nn.Linear(256, 128),
        )

        # ---------------- Numerical ----------------
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_num, 128), nn.BatchNorm1d(128, affine=False), nn.ReLU(),
            nn.Linear(128, 128),
        )

        # ---------------- Fusion → LSTM ----------------
        fused_len = 4 * 128
        self.vec_len = 128

        # Initial (h0, c0) encoders from fused vector
        self.hc_mlp = nn.Sequential(
            nn.Linear(fused_len, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(),
            nn.Linear(512, self.vec_len),
        )

        # Per-step temporal encoders (x_t) from the same fused vector
        self.x_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_len, 768), nn.BatchNorm1d(768, affine=False), nn.ReLU(),
                nn.Linear(768, self.vec_len),
            )
            for _ in range(self.seq_length)
        ])

        # Unrolled LSTM cells and per-step heads
        self.cells = nn.ModuleList([nn.LSTMCell(self.vec_len, self.vec_len) for _ in range(self.seq_length)])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vec_len * 2, self.vec_len), nn.BatchNorm1d(self.vec_len, affine=False), nn.ReLU(),
                nn.Linear(self.vec_len, self.vec_len // 2),  nn.BatchNorm1d(self.vec_len // 2, affine=False), nn.ReLU(),
                nn.Linear(self.vec_len // 2, 1),
            )
            for _ in range(self.seq_length)
        ])

    # ---------------- Categorical helpers ----------------
    @staticmethod
    def _norm_key(s: str) -> str:
        """Normalize category string for stable lookup."""
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _embed_cat(self, categories: Sequence[str], titles: Sequence[str]) -> Tensor:
        """
        Embed category and inferred language, then combine via element-wise product (Eq. 9).
        Returns a [B, 128] feature.

        categories: list[str] length B
        titles:     list[str] length B (for language ID)
        """
        # Map categories via normalized keys
        cate_ids = []
        for c in categories:
            idx = self._cate_vocab_norm.get(self._norm_key(c), self.cate_ooa_idx)
            cate_ids.append(idx)

        # Infer language for each title
        langs = []
        for t in titles:
            code = langid.classify(t)[0] if t else "OOA"
            langs.append(self.lang_vocab.get(code, 0))

        cate = self.cate_embed(torch.tensor(cate_ids, device=device))  # [B, 128]
        lang = self.lang_embed(torch.tensor(langs, device=device))     # [B, 128]
        cate = self.cate_mlp(cate)                                     # [B, 128]
        lang = self.lang_mlp(lang)                                     # [B, 128]

        feat = self.emb_mlp(cate * lang)                               # [B, 128]
        return feat

    # ---------------- Forward ----------------
    def forward(
        self,
        img: Tensor,                         # [B, 3, 224, 224]
        texts: Sequence[Sequence[str]],      # 5 lists, each length B (collate-transposed)
        meta: Tensor,                        # [B, 6]
        cat: Sequence[Sequence[str]],        # [categories(list[B]), titles(list[B])]
    ) -> Tuple[Tensor, None]:
        """
        Returns
        -------
        out : [B, T] popularity scores (clamped ≥ 0)
        aux : None (placeholder for API parity)
        """
        B = img.size(0)

        # ---- Visual ----
        with torch.no_grad():
            vis = self.img_feature(img).view(B, -1)  # [B, 2048]
        vis = self.img_mlp(vis)                      # [B, 128]

        # ---- Text ----
        stacked = _bert_features(texts, self.tok, self.bert)  # [B, 5, 768, 1]
        txt = self.text_conv(stacked)                         # [B, 768]
        txt = self.text_mlp(txt)                              # [B, 128]

        # ---- Numerical ----
        met = self.meta_mlp(meta)                             # [B, 128]

        # ---- Categorical (category + language-from-title) ----
        categories, titles = cat[0], cat[1]
        emb = self._embed_cat(categories, titles)             # [B, 128]

        # ---- Fuse ----
        fused = torch.cat([vis, emb, txt, met], dim=1)        # [B, 512]

        # ---- LSTM unroll with per-step heads ----
        h = self.hc_mlp(fused)                                # [B, 128]
        c = self.hc_mlp(fused)                                # [B, 128]
        out = torch.empty(B, self.seq_length, device=device)

        for i in range(self.seq_length):
            x = self.x_mlps[i](fused)                         # [B, 128]
            h, c = self.cells[i](x, (h, c))
            s = torch.cat([h, c], dim=1)                      # [B, 256]
            y = self.heads[i](s).squeeze(1)                   # [B]
            out[:, i] = y.clamp_min(0.0)                      # scores are non-negative

        return out, None
