# POSTER_V2 — How to Run

**Machine tested on:** Windows · NVIDIA GeForce RTX 5070 Laptop GPU  
**Expected result:** ~92% accuracy on RAF-DB test set (3 068 images)

> The model weights are **not included** in this folder (356 MB total).  
> Step 3 downloads them automatically with one command.

---

## Folder structure

```
poster_v2/
├── HOW_TO_RUN.md              ← you are here
├── download_weights.py        ← downloads checkpoint + backbones
├── inference_wrapper.py       ← main inference script
├── main.py                    ← training script (not needed for inference)
├── config.yaml                ← all paths and hyperparameters
├── requirements.txt           ← Python dependencies
│
├── models/                    ← model architecture
│   ├── PosterV2_7cls.py
│   ├── ir50.py
│   ├── mobilefacenet.py
│   └── vit_model.py
│
├── data_preprocessing/        ← SAM optimizer (used by training)
│   └── sam.py
│
├── checkpoint/                ← rafdb_best.pth lands here after Step 3
└── pretrain/                  ← ir50.pth and mobilefacenet land here after Step 3
```

---

## Step 1 — Create the conda environment

Open **Anaconda Prompt** and run:

```bat
conda create -n poster_v2 python=3.10 -y
conda activate poster_v2
```

---

## Step 2 — Install dependencies

Navigate to this folder first:

```bat
cd path\to\poster_v2
```

Install PyTorch with **CUDA 12.8** (required for RTX 5070):

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages:

```bat
pip install -r requirements.txt
```

Verify your GPU is visible:

```bat
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output: `NVIDIA GeForce RTX 5070 Laptop GPU`

---

## Step 3 — Download model weights  *(one-time, ~356 MB)*

```bat
python download_weights.py
```

This downloads three files from the official POSTER_V2 Google Drive links:

| File | Size | Description |
|---|---|---|
| `checkpoint/rafdb_best.pth` | 241 MB | POSTER_V2 trained on RAF-DB |
| `pretrain/ir50.pth` | 122 MB | IR-50 face recognition backbone |
| `pretrain/mobilefacenet_model_best.pth.tar` | 12 MB | MobileFaceNet landmark detector |

Already downloaded? The script detects existing files and skips them automatically.

---

## Step 4 — Fix package imports  *(Windows only, one-time)*

Windows sometimes drops empty `__init__.py` files during extraction.  
Re-create them to be safe:

```bat
echo. > models\__init__.py
echo. > data_preprocessing\__init__.py
```

---

## Step 5 — Run inference

```bat
python inference_wrapper.py --config config.yaml
```

### What you should see

```
Device: cuda
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Checkpoint: ...\checkpoint\rafdb_best.pth
Loaded MobileFaceNet weights from ...\pretrain\mobilefacenet_model_best.pth.tar
Loaded IR-50 weights from ...\pretrain\ir50.pth
Model loaded from ...\checkpoint\rafdb_best.pth
Class 1: 100%|██████████| 329/329
Class 2: 100%|██████████| 74/74
...
Inference complete.
  Images  : 3068
  Accuracy: 92.xx%
  CSV     : ...\predictions\poster_v2.csv
```

Output is written to `predictions/poster_v2.csv` at the project root.

---

## All steps as one block

```bat
conda activate poster_v2
cd path\to\poster_v2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
echo. > models\__init__.py
echo. > data_preprocessing\__init__.py
python download_weights.py
python inference_wrapper.py --config config.yaml
```

---

## Troubleshooting

| Error message | Fix |
|---|---|
| `No module named 'models'` | Re-run `echo. > models\__init__.py` (Step 4) |
| `No checkpoint found for pattern: checkpoint/rafdb_best.pth` | Re-run `python download_weights.py` (Step 3) |
| `CUDA: False` / running on CPU | PyTorch installed without CUDA. Reinstall: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `no kernel image is available for execution` | PyTorch version too old for RTX 5070. Must be ≥ 2.6.0. Reinstall as above. |
| `cannot import name 'trunc_normal_' from 'timm.models.layers'` | Upgrade timm: `pip install --upgrade "timm>=1.0.0"` |
| Download fails / Google Drive quota | Download manually from the links in `download_weights.py` and place files in `checkpoint/` and `pretrain/` |

---

## Data layout expected

The test images must be at:

```
data/raf_db/test/
├── 1/    329  images   (Surprise)
├── 2/    74   images   (Fear)
├── 3/    160  images   (Disgust)
├── 4/    1185 images   (Happiness)
├── 5/    478  images   (Sadness)
├── 6/    162  images   (Anger)
└── 7/    680  images   (Neutral)
```

Path is configured in `config.yaml` → `data_dir` + `valid_split`.

---

*CSCI323 Group Project — POSTER_V2 (Khanh)*
