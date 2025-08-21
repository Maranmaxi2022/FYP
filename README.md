This repository provides a strong multi-modal baseline built with a **Time-Series Transformer (TST)** as the temporal head (replacing the LSTM used in the [paper’s](https://arxiv.org/abs/2503.04446) original baseline).
The model fuses **visual** (thumbnail), **textual** (mBERT), **categorical** (category + language), and **numerical** features (plus optional **Early Popularity, EP**), and predicts a sequence of daily popularity scores.

<p align="center">
  <img src="src/FYP.png" width="820" alt="Model overview">
</p>

## Data format

* **CSV** (e.g., `basic_view_pn.csv`) and an **image folder** `img_yt/` containing `<video_id>.jpg` thumbnails.

**Columns (0‑indexed):**

* `0`  - `video_id`
* `5`  - `user_id`
* `6`  - `category` (one of 15 YouTube categories in English)
* `7`  - `title`
* `8`  - `keywords/hashtags`
* `9`  - `description`
* `10..14` - five numeric features (floats; e.g., duration, followers, posts, title\_len, tag\_count)
* `15..` - **popularity score sequence (Day‑1..Day‑30)**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

### 1) TST **without** EP → predicts **Days 1–30** (30 steps)

```bash
python scripts/main.py --train --temporal tst \
  --K_fold 0 --seq_len 30 \
  --data_files data/basic_view_pn.csv \
  --images_dir data/img_yt \
  --ckpt_path ckpts/tst_wo_ep
```

### 2) TST **with** EP → predicts **Days 2–30** (29 steps)

```bash
python scripts/main.py --train --temporal tst \
  --K_fold 0 --use_ep --seq_len 29 \
  --data_files data/basic_view_pn.csv \
  --images_dir data/img_yt \
  --ckpt_path ckpts/tst_w_ep
```

## Evaluation

```bash
# Example: TST without EP, window 1..30
python scripts/main.py --eval_all --temporal tst \
  --K_fold 0 --seq_len 30 \
  --data_files data/basic_view_pn.csv --images_dir data/img_yt \
  --ckpt_path ckpts/tst_wo_ep \
  --eval_from 1 --eval_to 30
```

```bash
# Example: TST with EP, window 1..29 (corresponds to calendar Days 2..30)
python scripts/main.py --eval_all --temporal tst \
  --K_fold 0 --use_ep --seq_len 29 \
  --data_files data/basic_view_pn.csv --images_dir data/img_yt \
  --ckpt_path ckpts/tst_w_ep \
  --eval_from 1 --eval_to 29
```

## Results

### Table I — Component study (✓ marks enabled modules)

| BERT-Base | BERT-Mul | MLP | LSTM | **TST** |  **AMAE** |  **ASRC** |
| :-------: | :------: | :-: | :--: | :-----: | --------: | --------: |
|     ✓     |          |     |   ✓  |         |     0.782 |     0.958 |
|           |     ✓    |  ✓  |      |         |     0.786 |     0.958 |
|           |     ✓    |     |   ✓  |         |     0.746 |     0.958 |
|           |     ✓    |     |      |  **✓**  | **0.723** | **0.960** |

### Table II — SMTPD Day-7 / Day-14 / Day-30 (MAE/SRC averages)

| Method                                                        |  SMTPD (day 7)  |  SMTPD (day 14) |  SMTPD (day 30) |
|:--------------------------------------------------------------| :-------------: | :-------------: | :-------------: |
| [Ding et al.](https://dl.acm.org/doi/10.1145/3343031.3356062) |   1.715/0.849   |   1.669/0.846   |   1.592/0.843   |
|  w. EP                                                        |   0.715/0.964   |   0.742/0.959   |   0.749/0.931   |
| [Lai et al.](https://dl.acm.org/doi/10.1145/3394171.3416273)  |   1.573/0.875   |   1.524/0.872   |   1.495/0.864   |
|  w. EP                                                        |   0.725/0.957   |   0.753/0.962   |   0.760/0.957   |
| [Xu et al.](https://dl.acm.org/doi/10.1145/3394171.3416274)   |   1.895/0.817   |   1.832/0.818   |   1.743/0.820   |
|  w. EP                                                        |   0.754/0.962   |   0.798/0.956   |   0.822/0.949   |
| [SMTPD (Xu et al.)](https://arxiv.org/abs/2503.04446)         |   1.673/0.852   |   1.628/0.850   |   1.563/0.848   |
|  w. EP                                                        |   0.732/0.963   |   0.766/0.957   |   0.775/0.952   |
| **Ours w/o. EP**                                              | **1.651/0.856** | **1.605/0.854** | **1.542/0.851** |
| **Ours**                                                      | **0.710/0.964** | **0.737/0.959** | **0.745/0.954** |

### Table III — EP assessment across scenarios

| Method           |  **AMAE** |  **ASRC** | **MAE** (day 30 only) | **SRC** (day 30 only) |
| :--------------- | --------: | --------: |----------------------:|----------------------:|
| w/o. EP (1–30)   |     1.540 |     0.866 |                 1.530 |                 0.850 |
| w/o. EP (2–30)   |     1.610 |     0.852 |                 1.551 |                 0.849 |
| **Ours (w. EP)** | **0.723** | **0.960** |                 0.741 |                 0.954 |