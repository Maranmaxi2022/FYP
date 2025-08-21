This repo contains our research code and instructions to reproduce the **SMTPD** baseline results reported in the paper (https://arxiv.org/abs/2503.04446).

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

## Data format

* **CSV** (e.g., `basic_view_pn.csv`) and an **image folder** `img_yt/` containing `<video_id>.jpg` thumbnails.

**Columns (0‑indexed):**

* `0`  — `video_id`
* `5`  — `user_id`
* `6`  — `category` (one of 15 YouTube categories in English)
* `7`  — `title`
* `8`  — `keywords/hashtags`
* `9`  — `description`
* `10..14` — five numeric features (floats; e.g., duration, followers, posts, title\_len, tag\_count)
* `15..` — **popularity score sequence (Day‑1..Day‑30)** in the paper’s log unit `p = log2(v/d + 1)`

## Model summary

Implemented in `smp_model.py`:

* **Visual**: ResNet‑101 backbone → 2048 → MLP → **128‑D**
* **Text** (5 fields via mBERT): pooled outputs → stack → **1×1 conv** across fields → MLP → **128‑D**
* **Numerical**: 5 stats + **EP (optional Day‑1)** → MLP → **128‑D**
* **Categorical**: category embedding + language‑from‑title (langid) → MLPs → element‑wise product → **128‑D**
* **Fusion**: concat 4×128 → **512‑D** → LSTMCell unrolled for T steps → per‑step regression heads
* **Loss**: Composite Gradient Loss (SmoothL1 + Δ/Δ² + peak alignment + tiny Laplacian; cosine‑annealed weights)

## Training

We train **two models** (same architecture, different config).

### 1) **No‑EP model** (predicts Days **1–30**)

```bash
DATA=/path/to/basic_view_pn.csv
IMG=/path/to/img_yt
CK=./ckpts_no_ep

python main.py --train --K_fold 0 --seq_len 30 \
  --data_files "$DATA" --images_dir "$IMG" --ckpt_path "$CK"
```

### 2) **With‑EP model (“Ours”)** (EP=Day‑1 input; predicts Days **2–30**, 29 steps)

```bash
CK=./ckpts_with_ep

python main.py --train --K_fold 0 --seq_len 29 --use_ep \
  --data_files "$DATA" --images_dir "$IMG" --ckpt_path "$CK"
```

**Defaults:** `--epochs 20`, `--batch_size 64`. Use `--seed` for reproducibility and `--no_ckpt` to skip saving.

## Selecting the best epoch

Each training saves `<K>-epoch<N>.pth`. Use `--eval_all` to score every epoch and print a leaderboard by **AMAE** on a chosen window.

**No‑EP (1–30 window):**

```bash
python main.py --eval_all --K_fold 0 --seq_len 30 \
  --eval_from 1 --eval_to 30 \
  --data_files "$DATA" --images_dir "$IMG" --ckpt_path "$CK"
```

**With‑EP (2–30 window = steps 1..29):**

```bash
python main.py --eval_all --K_fold 0 --seq_len 29 --use_ep \
  --eval_from 1 --eval_to 29 \
  --data_files "$DATA" --images_dir "$IMG" --ckpt_path "$CK"
```

Note the top checkpoint name (e.g., `0-epoch12.pth`).

## EP setups

1. **w/o. EP (1–30)** — No‑EP model, window **1..30**

```bash
python main.py --test --K_fold 0 --seq_len 30 \
  --ckpt_path ./ckpts_no_ep --ckpt_name <BEST_NOEP>.pth \
  --eval_from 1 --eval_to 30 \
  --data_files "$DATA" --images_dir "$IMG"
```

2. **w/o. EP (2–30)** — same No‑EP checkpoint, window **2..30**

```bash
python main.py --test --K_fold 0 --seq_len 30 \
  --ckpt_path ./ckpts_no_ep --ckpt_name <BEST_NOEP>.pth \
  --eval_from 2 --eval_to 30 \
  --data_files "$DATA" --images_dir "$IMG"
```

3. **Ours (with EP)** — With‑EP model, window **1..29** (true Days 2–30)

```bash
python main.py --test --K_fold 0 --seq_len 29 --use_ep \
  --ckpt_path ./ckpts_with_ep --ckpt_name <BEST_EP>.pth \
  --eval_from 1 --eval_to 29 \
  --data_files "$DATA" --images_dir "$IMG"
```