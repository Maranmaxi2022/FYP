"""
main.py — SMTPD trainer/evaluator with LSTM or TST temporal head

Paper: https://arxiv.org/abs/2503.04446

What this script does
---------------------
• Loads the SMTPD multi-modal dataset (covers, text, numeric, categorical).
• Builds the model backbone (ResNet-101 + mBERT encoders + numeric/categorical MLPs).
• Lets you choose the temporal head:
    - LSTM baseline  (unrolled LSTMCell with per-day heads)
    - TST (Time-series Transformer)  ← default for your current approach
• Trains/evaluates with Composite Gradient Loss (SmoothL1 + 1st/2nd diffs + peak alignment).
• Supports windowed evaluation (e.g., Table-5 style 1–30 or 2–30, day-30 only, etc.).

Key data conventions (very important)
-------------------------------------
• Labels in the CSV are popularity scores for Day-1 .. Day-30 (not raw views).
• EP (Early Popularity) is **not a separate column**. EP = **Day-1** label.
• If you pass `--use_ep`, the dataloader:
    - adds Day-1 as an extra numeric input,
    - shifts targets to **Days 2..** to avoid leakage.
  Typical horizons:
    - With EP:  use `--seq_len 29`  (predict Days 2..30)
    - Without:  use `--seq_len 30`  (predict Days 1..30)

Quick start
-----------
Train TST (fold 0), no-EP (predict 1..30):
    python scripts/main.py --train --temporal tst --K_fold 0 \
      --seq_len 30 --data_files /path/to/basic_view_pn.csv \
      --images_dir /path/to/img_yt --ckpt_path ./ckpts/tst_noep

Train TST with EP (EP=Day-1, predict 2..30):
    python scripts/main.py --train --temporal tst --use_ep --K_fold 0 \
      --seq_len 29 --data_files /path/to/basic_view_pn.csv \
      --images_dir /path/to/img_yt --ckpt_path ./ckpts/tst_ep

Evaluate all checkpoints on a specific window (e.g., Table-5):
    # AMAE/ASRC over days 1..30
    python scripts/main.py --eval_all --temporal tst --K_fold 0 \
      --seq_len 30 --eval_from 1 --eval_to 30 --ckpt_path ./ckpts/tst_noep

    # AMAE/ASRC over days 2..30 (EP setting)
    python scripts/main.py --eval_all --temporal tst --K_fold 0 \
      --seq_len 29 --eval_from 1 --eval_to 29 --ckpt_path ./ckpts/tst_ep

    # Day-30 only (MAE/SRC)
    python scripts/main.py --eval_all --temporal tst --K_fold 0 \
      --seq_len 30 --eval_from 30 --eval_to 30 --ckpt_path ./ckpts/tst_noep
"""

import os
import csv
import math
import argparse
import warnings
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from smp_model import youtube_lstm3, youtube_tst3  # ← includes both heads
from smp_data import youtube_data_lstm
from tools import print_output_seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- CLI ---------------------------------
parser = argparse.ArgumentParser(description="SMTPD (LSTM/TST)")

# Training setup
parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
parser.add_argument('--seed', type=int, default=23, help='Random seed')

# Data paths
parser.add_argument('--images_dir', type=str, default='/workspace/SMTPD_project/data/img_yt',
                    help='Folder with cover images named <video_id>.jpg')
parser.add_argument('--gt_path', type=str, default='0', help='(Unused placeholder)')
parser.add_argument('--data_files', type=str, default='/workspace/SMTPD_project/data/basic_view_pn.csv',
                    help='Path to the CSV with multi-modal features and labels')

# Sequence + outputs
parser.add_argument('--seq_len', type=int, default=30,
                    help='Prediction horizon (days). With EP use 29; without EP use 30.')
parser.add_argument('--ckpt_path', type=str, default='/workspace/SMTPD_project/ckpts',
                    help='Where to save/load checkpoints')
parser.add_argument('--result_file', type=str, default='results.csv',
                    help='CSV to write test predictions to (saved under ckpt_path)')
parser.add_argument('--no_ckpt', action='store_true', help='Disable checkpoint saving')

# Modes / options
parser.add_argument('--train', action='store_true', help='Train mode')
parser.add_argument('--test', action='store_true', help='Test mode (requires a checkpoint)')
parser.add_argument('--K_fold', type=int, default=0, help='Fold index in [0..4] (5-fold split)')
parser.add_argument('--use_ep', action='store_true',
                    help='Use EP=Day-1 as extra numeric input and shift labels to Days 2..')

# Evaluation helpers
parser.add_argument('--ckpt_name', type=str, default='', help='Checkpoint to test, e.g., 0-epoch10.pth')
parser.add_argument('--eval_all', action='store_true',
                    help='Evaluate every <K>-epoch*.pth in --ckpt_path on the val split')
parser.add_argument('--eval_from', type=int, default=1,
                    help='First day (1-indexed) for metrics window; default=1')
parser.add_argument('--eval_to', type=int, default=0,
                    help='Last day (1-indexed) for metrics window; 0=seq_len')

# Temporal head selector (NEW) + TST hparams
parser.add_argument('--temporal', choices=['lstm', 'tst'], default='tst',
                    help='Sequence model: LSTM baseline or Transformer (TST)')
parser.add_argument('--tst_d_model', type=int, default=256)
parser.add_argument('--tst_nhead', type=int, default=8)
parser.add_argument('--tst_layers', type=int, default=4)
parser.add_argument('--tst_ff', type=int, default=512)
parser.add_argument('--tst_dropout', type=float, default=0.1)


# ---------------------- Utilities ----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _slice_days(labels, preds, start_day: int, end_day: int, T: int):
    """
    Slice [N, T] labels/preds into a 1-indexed window [start_day, end_day].
    If end_day==0, use T (full length).
    """
    L = np.asarray(labels)
    P = np.asarray(preds)
    if end_day == 0:
        end_day = T
    s = max(1, start_day) - 1
    e = min(T, end_day)
    return L[:, s:e], P[:, s:e]


# ---------------------- Composite Gradient Loss ---------------------
class CompositeGradientLoss(nn.Module):
    """
    Composite Gradient Loss (CGL).
    Terms: SmoothL1 on sequence + SmoothL1 on Δ and Δ² + peak-day alignment (L1 on one-hot argmax).
    """
    def __init__(self, beta: float = 0.1, lambda1: float = 1.0, lambda2: float = 1.0,
                 alpha: float = 1.0):
        super().__init__()
        self.base = nn.SmoothL1Loss(beta)
        self.l1 = lambda1
        self.l2 = lambda2
        self.alpha = alpha

    @staticmethod
    def _anneal(step: int, total: int) -> float:
        # Cosine schedule in [0,1]
        total = max(total, 1)
        return 0.5 * (1.0 + math.cos(step / total * math.pi))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                step: int, total_steps: int) -> torch.Tensor:
        loss = self.base(pred, target)
        d1_p, d1_t = pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1]
        d2_p, d2_t = d1_p[:, 1:] - d1_p[:, :-1], d1_t[:, 1:] - d1_t[:, :-1]
        k = self._anneal(step, total_steps)
        loss = loss + k * self.l1 * self.base(d1_p, d1_t)
        loss = loss + k * self.l2 * self.base(d2_p, d2_t)
        peak_p = torch.argmax(pred, dim=1)
        peak_t = torch.argmax(target, dim=1)
        oh_p = torch.nn.functional.one_hot(peak_p, num_classes=pred.size(1)).float()
        oh_t = torch.nn.functional.one_hot(peak_t, num_classes=pred.size(1)).float()
        loss = loss + k * self.alpha * torch.nn.functional.l1_loss(oh_p, oh_t)
        return loss


# --------------------------- Data split -----------------------------
def _split_kfold(ds: youtube_data_lstm, K: int, fold_idx: int,
                 batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Reproducible 5-fold split:
      - val = contiguous 1/K slice
      - train = remaining data
      - test  = same as val (paper-style)
    """
    rng = random.Random(23)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    fold = len(ds) // K if K > 0 else len(ds)
    v0, v1 = fold_idx * fold, (fold_idx + 1) * fold
    val_idx = idx[v0:v1] if K > 0 else idx[:fold]
    train_idx = [i for i in idx if i not in val_idx]

    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)
    test_set = Subset(ds, val_idx)  # test == val

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=True)
    return train_loader, val_loader, test_loader


# ---------------------------- Loops --------------------------------
def train_loop(args: argparse.Namespace, model: nn.Module,
               train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Standard supervised training loop with CGL and ReduceLROnPlateau scheduler."""
    loss_fn = CompositeGradientLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=1, min_lr=0.0)

    os.makedirs(args.ckpt_path, exist_ok=True)
    total_steps = len(train_loader) * args.epochs
    val_crit = nn.SmoothL1Loss(beta=0.1)  # Validation proxy

    for ep in range(args.epochs):
        # ---- Train ----
        model.train()
        tr_losses: List[float] = []
        preds, labels = [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {ep+1}")):
            img  = batch['img'].to(device)
            text = batch['text']
            meta = batch['meta'].to(device)
            cat  = batch['cat']
            y    = batch['label'].to(device)

            yhat, _ = model(img, text, meta, cat)
            loss = loss_fn(yhat, y, ep * len(train_loader) + step, total_steps)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            tr_losses.append(float(loss.item()))
            preds.extend(yhat.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

        print(f"Epoch {ep+1} train loss: {sum(tr_losses)/len(tr_losses):.6f}")
        print_output_seq(labels, preds)

        # ---- Validate ----
        model.eval()
        v_losses: List[float] = []
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                img  = batch['img'].to(device)
                text = batch['text']
                meta = batch['meta'].to(device)
                cat  = batch['cat']
                y    = batch['label'].to(device)
                yhat, _ = model(img, text, meta, cat)
                v_losses.append(float(val_crit(yhat, y).item()))
                v_preds.extend(yhat.cpu().numpy().tolist())
                v_labels.extend(y.cpu().numpy().tolist())

        val_loss = sum(v_losses) / len(v_losses) if v_losses else 0.0
        print(f"Epoch {ep+1} val loss: {val_loss:.6f}")

        # Windowed metrics (e.g., 1–30, 2–30, 30–30)
        Lwin, Pwin = _slice_days(v_labels, v_preds, args.eval_from, args.eval_to, args.seq_len)
        print(f"[Eval window] days {args.eval_from}..{args.eval_to or args.seq_len}")
        print_output_seq(Lwin, Pwin)

        sch.step(val_loss)

        if not args.no_ckpt:
            ck = os.path.join(args.ckpt_path, f"{args.K_fold}-epoch{ep+1}.pth")
            torch.save(model.state_dict(), ck)


def test_loop(args: argparse.Namespace, model: nn.Module, loader: DataLoader) -> None:
    """Evaluate a single checkpoint; writes per-sample predictions to CSV and prints metrics."""
    model.eval()
    out_path = os.path.join(args.ckpt_path, args.result_file)
    open(out_path, 'w').close()

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            img  = batch['img'].to(device)
            text = batch['text']
            meta = batch['meta'].to(device)
            cat  = batch['cat']
            y    = batch['label'].to(device)
            yhat, _ = model(img, text, meta, cat)

            preds.extend(yhat.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

            with open(out_path, 'a+', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                for i in range(yhat.size(0)):
                    p_i = yhat[i].detach().cpu().numpy().tolist()
                    l_i = y[i].detach().cpu().numpy().tolist()
                    w.writerow([batch['id'][i], p_i, l_i])

    Lwin, Pwin = _slice_days(labels, preds, args.eval_from, args.eval_to, args.seq_len)
    print(f"[Test window] days {args.eval_from}..{args.eval_to or args.seq_len}")
    print_output_seq(Lwin, Pwin)


# --------------- Evaluate all checkpoints for a fold ---------------
def eval_all_checkpoints(args: argparse.Namespace, model: nn.Module,
                         val_loader: DataLoader) -> None:
    """
    Loads every '<K>-epoch*.pth' in args.ckpt_path, evaluates on val_loader
    over the selected day window, prints a leaderboard, and writes scores_fold<K>.csv.
    """
    ckpts = sorted(Path(args.ckpt_path).glob(f"{args.K_fold}-epoch*.pth"))
    if not ckpts:
        print(f"No checkpoints found in: {args.ckpt_path}")
        return

    rows = []
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Eval {ck.name}"):
                img  = batch['img'].to(device)
                text = batch['text']
                meta = batch['meta'].to(device)
                cat  = batch['cat']
                y    = batch['label'].to(device)
                yhat, _ = model(img, text, meta, cat)
                preds.extend(yhat.cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().tolist())

        Lwin, Pwin = _slice_days(labels, preds, args.eval_from, args.eval_to, args.seq_len)
        _, _, _, AMAE, _, ASRC = print_output_seq(Lwin, Pwin)
        rows.append((ck.name, AMAE, ASRC))

    rows.sort(key=lambda r: r[1])  # by AMAE asc
    print(f"\n=== Leaderboard (by AMAE, lower is better) — window {args.eval_from}..{args.eval_to or args.seq_len} ===")
    for name, amae, asrc in rows:
        print(f"{name:>15}  AMAE={amae:.3f}  ASRC={asrc:.3f}")

    out_csv = Path(args.ckpt_path) / f"scores_fold{args.K_fold}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "AMAE", "ASRC", "window_from", "window_to"])
        for name, amae, asrc in rows:
            w.writerow([name, amae, asrc, args.eval_from, (args.eval_to or args.seq_len)])
    print(f"\nWrote: {out_csv}")


# ------------------------------ Main -------------------------------
def main() -> None:
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    set_seed(args.seed)

    # Dataset + loaders
    ds = youtube_data_lstm(args.data_files, args.images_dir, args.gt_path)
    ds.seq_len = args.seq_len
    ds.p_i = 1 if args.use_ep else 0  # include EP (Day-1) if requested; labels shift to Days 2..

    train_loader, val_loader, test_loader = _split_kfold(
        ds, K=5, fold_idx=args.K_fold, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model: pick temporal head
    if args.temporal == 'tst':
        model = youtube_tst3(
            args.seq_len, args.batch_size,
            d_model=args.tst_d_model, nhead=args.tst_nhead,
            num_layers=args.tst_layers, dim_ff=args.tst_ff,
            dropout=args.tst_dropout
        ).to(device)
    else:
        model = youtube_lstm3(args.seq_len, args.batch_size).to(device)

    if args.eval_all:
        eval_all_checkpoints(args, model, val_loader)
        return

    if args.test:
        ck = os.path.join(args.ckpt_path, args.ckpt_name) if args.ckpt_name \
             else os.path.join(args.ckpt_path, f"{args.K_fold}-epoch1.pth")
        model.load_state_dict(torch.load(ck, map_location=device))
        print("Loaded:", ck)
        if args.result_file == 'results.csv':
            args.result_file = f"results_{Path(ck).stem}.csv"
        test_loop(args, model, test_loader)
        return

    if args.train:
        train_loop(args, model, train_loader, val_loader)
    else:
        print("Choose one: --train or --test or --eval_all")


if __name__ == "__main__":
    main()
