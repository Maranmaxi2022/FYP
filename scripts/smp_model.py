"""
smp_model.py — SMTPD multi-modal sequence regressors

Encoders (shared by both heads)
-------------------------------
• Visual (cover image): ResNet-101 backbone → 2048-D → MLP → 128-D
• Textual (5 fields: user_id, category, title, keywords, description):
    mBERT pooled outputs (each 768-D) → stack → 1×1 conv over fields → 768-D → MLP → 128-D
• Numerical (5 features + optional EP=Day-1): MLP → 128-D
• Categorical (category + language inferred from title):
    category embedding (128) & language embedding (128) → MLPs → (⊙) → MLP → 128-D
Fusion: concat [visual, categorical, textual, numeric] → 512-D

Temporal heads
--------------
1) youtube_lstm3  — baseline unrolled LSTMCell with per-day heads (existing).
2) youtube_tst3   — Time-series Transformer (TST): fused vector → linear to d_model,
   tile to T tokens → add positional encodings → TransformerEncoder → per-day MLP head.

Both heads output a [B, T] tensor of non-negative popularity scores.
"""

from __future__ import annotations

import os
import re
import math
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
            (PyTorch default_collate transposes lists this way.)
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
# LSTM temporal head (unchanged)
# -------------------------------------------------------------------
class youtube_lstm3(nn.Module):
    """
    SMTPD baseline multi-modal sequence regressor using an unrolled LSTM.
    Produces non-negative scores of shape [B, T].
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
        def _norm_key(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        self._cate_vocab_norm = {
            _norm_key("People & Blogs"):1, _norm_key("Gaming"):2, _norm_key("News & Politics"):3,
            _norm_key("Entertainment"):4, _norm_key("Music"):5, _norm_key("Education"):6,
            _norm_key("Sports"):7, _norm_key("Howto & Style"):8, _norm_key("Film & Animation"):9,
            _norm_key("Nonprofits & Activism"):10, _norm_key("Travel"):11, _norm_key("Comedy"):12,
            _norm_key("Science & Technology"):13, _norm_key("Autos & Vehicles"):14, _norm_key("Pets & Animals"):15,
        }
        self.cate_ooa_idx = 0
        self.lang_vocab = {"en":1, "zh":2, "ko":3, "ja":4, "hi":5, "ru":6, "OOA":0}

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

        self.hc_mlp = nn.Sequential(
            nn.Linear(fused_len, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(),
            nn.Linear(512, self.vec_len),
        )

        self.x_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_len, 768), nn.BatchNorm1d(768, affine=False), nn.ReLU(),
                nn.Linear(768, self.vec_len),
            )
            for _ in range(self.seq_length)
        ])

        self.cells = nn.ModuleList([nn.LSTMCell(self.vec_len, self.vec_len) for _ in range(self.seq_length)])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vec_len * 2, self.vec_len), nn.BatchNorm1d(self.vec_len, affine=False), nn.ReLU(),
                nn.Linear(self.vec_len, self.vec_len // 2),  nn.BatchNorm1d(self.vec_len // 2, affine=False), nn.ReLU(),
                nn.Linear(self.vec_len // 2, 1),
            )
            for _ in range(self.seq_length)
        ])

    @staticmethod
    def _norm_key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _embed_cat(self, categories: Sequence[str], titles: Sequence[str]) -> Tensor:
        cate_ids = [self._cate_vocab_norm.get(self._norm_key(c), self.cate_ooa_idx) for c in categories]
        langs = [langid.classify(t)[0] if t else "OOA" for t in titles]
        langs = [ {"en":1,"zh":2,"ko":3,"ja":4,"hi":5,"ru":6}.get(l,0) for l in langs ]

        cate = self.cate_embed(torch.tensor(cate_ids, device=device))
        lang = self.lang_embed(torch.tensor(langs, device=device))
        cate = self.cate_mlp(cate)
        lang = self.lang_mlp(lang)
        return self.emb_mlp(cate * lang)

    def forward(self, img: Tensor, texts: Sequence[Sequence[str]], meta: Tensor,
                cat: Sequence[Sequence[str]]) -> Tuple[Tensor, None]:
        B = img.size(0)

        with torch.no_grad():
            vis = self.img_feature(img).view(B, -1)
        vis = self.img_mlp(vis)

        stacked = _bert_features(texts, self.tok, self.bert)
        txt = self.text_conv(stacked)
        txt = self.text_mlp(txt)

        met = self.meta_mlp(meta)

        categories, titles = cat[0], cat[1]
        emb = self._embed_cat(categories, titles)

        fused = torch.cat([vis, emb, txt, met], dim=1)  # [B, 512]

        h = self.hc_mlp(fused)
        c = self.hc_mlp(fused)
        out = torch.empty(B, self.seq_length, device=device)
        for i in range(self.seq_length):
            x = self.x_mlps[i](fused)
            h, c = self.cells[i](x, (h, c))
            s = torch.cat([h, c], dim=1)
            y = self.heads[i](s).squeeze(1)
            out[:, i] = y.clamp_min(0.0)
        return out, None


# -------------------------------------------------------------------
# TST temporal head (NEW)
# -------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encodings."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: Tensor) -> Tensor:  # x: [B, L, D]
        return x + self.pe[:, : x.size(1)]


class youtube_tst3(nn.Module):
    """
    Transformer-based temporal head (“Time-series Transformer”, TST).

    Pipeline:
      fused(512) → Linear(d_model) → tile to T tokens → +PosEnc → TransformerEncoder →
      per-token MLP → [B, T] (non-negative scores).
    """
    def __init__(self, seq_length: int, batch_size: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.text_num = 5
        self.meta_num = 6

        # --- Shared encoders (same as LSTM head) ---
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.img_feature = nn.Sequential(*list(backbone.children())[:-1])
        self.img_mlp = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512,  affine=False), nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.tok, self.bert = _bert()
        self.text_conv = _TextConv(text_num=self.text_num)
        self.text_mlp = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(),
            nn.Linear(512, 128),
        )

        def _norm_key(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        self._cate_vocab_norm = {
            _norm_key("People & Blogs"):1, _norm_key("Gaming"):2, _norm_key("News & Politics"):3,
            _norm_key("Entertainment"):4, _norm_key("Music"):5, _norm_key("Education"):6,
            _norm_key("Sports"):7, _norm_key("Howto & Style"):8, _norm_key("Film & Animation"):9,
            _norm_key("Nonprofits & Activism"):10, _norm_key("Travel"):11, _norm_key("Comedy"):12,
            _norm_key("Science & Technology"):13, _norm_key("Autos & Vehicles"):14, _norm_key("Pets & Animals"):15,
        }
        self.cate_ooa_idx = 0
        self.lang_vocab = {"en":1, "zh":2, "ko":3, "ja":4, "hi":5, "ru":6, "OOA":0}
        self.cate_embed = nn.Embedding(16, 128)
        self.lang_embed = nn.Embedding(7, 128)
        self.cate_mlp = nn.Sequential(nn.Linear(128,256), nn.BatchNorm1d(256, affine=False), nn.ReLU(), nn.Linear(256,128))
        self.lang_mlp = nn.Sequential(nn.Linear(128,256), nn.BatchNorm1d(256, affine=False), nn.ReLU(), nn.Linear(256,128))
        self.emb_mlp  = nn.Sequential(nn.Linear(128,256), nn.BatchNorm1d(256, affine=False), nn.ReLU(), nn.Linear(256,128))

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_num, 128), nn.BatchNorm1d(128, affine=False), nn.ReLU(),
            nn.Linear(128, 128),
        )

        # --- Fusion → Transformer ---
        fused_len = 4 * 128  # 512
        self.fuse2model = nn.Linear(fused_len, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_length)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _norm_key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _embed_cat(self, categories: Sequence[str], titles: Sequence[str]) -> Tensor:
        cate_ids = [self._cate_vocab_norm.get(self._norm_key(c), self.cate_ooa_idx) for c in categories]
        langs = [langid.classify(t)[0] if t else "OOA" for t in titles]
        langs = [ {"en":1,"zh":2,"ko":3,"ja":4,"hi":5,"ru":6}.get(l,0) for l in langs ]
        cate = self.cate_embed(torch.tensor(cate_ids, device=device))
        lang = self.lang_embed(torch.tensor(langs, device=device))
        cate = self.cate_mlp(cate); lang = self.lang_mlp(lang)
        return self.emb_mlp(cate * lang)

    def forward(self, img: Tensor, texts: Sequence[Sequence[str]], meta: Tensor,
                cat: Sequence[Sequence[str]]) -> Tuple[Tensor, None]:
        B = img.size(0)

        # Encoders (frozen convnet for memory; BERT pooled features)
        with torch.no_grad():
            vis = self.img_feature(img).view(B, -1)
        vis = self.img_mlp(vis)

        stacked = _bert_features(texts, self.tok, self.bert)
        txt = self.text_conv(stacked)
        txt = self.text_mlp(txt)

        met = self.meta_mlp(meta)

        categories, titles = cat[0], cat[1]
        emb = self._embed_cat(categories, titles)

        fused = torch.cat([vis, emb, txt, met], dim=1)        # [B, 512]

        # Transformer over time: tile fused vector to T tokens, add PE, encode, project
        z = self.fuse2model(fused).unsqueeze(1)               # [B, 1, D]
        z = z.repeat(1, self.seq_length, 1)                   # [B, T, D]
        z = self.pos(z)
        z = self.encoder(z)                                   # [B, T, D]
        y = self.head(z).squeeze(-1)                          # [B, T]
        return y.clamp_min(0.0), None
