# LibreFace Fine-tuning on RAF-DB — How to Run

This folder contains two LibreFace workflows:

1. **Pretrained AffectNet baseline**
   - Uses the original LibreFace AffectNet checkpoint.
   - Runs inference on RAF-DB without RAF-DB fine-tuning.
   - Script: `inference.py`
   - Instructions: `Instructions.md`

2. **RAF-DB fine-tuned LibreFace**
   - Starts from the pretrained AffectNet checkpoint.
   - Replaces the final classifier with a 7-class RAF-DB classifier.
   - Trains/fine-tunes on RAF-DB.
   - Scripts: `train.py` and `inference_finetuned.py`
   - See: `instructions_finetune.md`
---

## Purpose

This workflow fine-tunes LibreFace on RAF-DB.

It starts from the pretrained LibreFace AffectNet checkpoint:

```text
models/libreface/checkpoints_fm_resnet/AffectNet/resnet.pt
```

Then it trains a new 7-class RAF-DB classifier and saves:

```text
models/libreface/checkpoints_rafdb/libreface_rafdb_best.pt
```

Final output:

```text
predictions/libreface.csv
```

Project labels:

```text
1 = Surprise
2 = Fear
3 = Disgust
4 = Happiness
5 = Sadness
6 = Anger
7 = Neutral
```

During training, labels are internally converted to `0..6` because PyTorch `CrossEntropyLoss` requires zero-based class indices. During inference, predictions are converted back to `1..7`.

---

## Environment setup

Open Anaconda Prompt or VS Code terminal with the correct environment:

```bash
conda activate libreface_test
```

Install required packages if missing:

```bash
pip install torch torchvision pandas numpy pillow tqdm matplotlib seaborn scikit-learn
```

For RTX 5070 GPU, use CUDA 12.8 PyTorch:

```bash
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); t=torch.tensor([2.0]).cuda(); print(t*t)"
```

Expected: CUDA is `True`, capability is `(12, 0)`, and tensor is on `cuda:0`.

---

## Step 1 — Train / fine-tune LibreFace

Go to the LibreFace folder:

```bash
cd "C:\Users\votah\Downloads\Minh Quan Dang - CSCI323 Group Project\models\libreface"
```

Run training:

```bash
python train.py --device cuda --batch-size 64 --epochs 100 --lr 1e-4 --freeze-encoder-epochs 3
```

Meaning:

```text
Epochs 1–3: train only the new RAF-DB classifier
Epoch 4 onward: unfreeze encoder and fine-tune the full model
```

Best checkpoint is saved automatically:

```text
models/libreface/checkpoints_rafdb/libreface_rafdb_best.pt
```

If CUDA memory error happens, reduce batch size:

```bash
python train.py --device cuda --batch-size 32 --epochs 30 --lr 1e-4 --freeze-encoder-epochs 3
```

---

## Step 2 — Run fine-tuned inference

After training finishes, run:

```bash
python inference_finetuned.py --split test --device cuda --batch-size 64
```

This uses:

```text
models/libreface/checkpoints_rafdb/libreface_rafdb_best.pt
```

and creates:

```text
predictions/libreface.csv
```

---

## Step 3 — Generate result tables and confusion matrix

Go back to the project root:

```bash
cd ..\..
```

Run:

```bash
python scripts\generate_results.py
```

This creates or updates:

```text
results/tables/model_comparison.csv
results/tables/per_class_f1.csv
results/plots/cm_libreface.png
```

---

## Step 4 — Optional consolidation check

If all three model prediction files exist:

```text
predictions/simple_cnn.csv
predictions/poster_v2.csv
predictions/libreface.csv
```

run:

```bash
python scripts\consolidate.py
```

## Important notes

Use `inference_finetuned.py` after training.

Do not use the old `inference.py` for the final fine-tuned result.

```text
inference.py
= pretrained AffectNet baseline only

inference_finetuned.py
= RAF-DB fine-tuned LibreFace result
```

In the report, describe the final method as:

```text
LibreFace was initialised from the pretrained AffectNet ResNet checkpoint.
The final 8-class classifier was replaced with a 7-class RAF-DB classifier.
The model was fine-tuned on RAF-DB train/val data and evaluated on the RAF-DB test set.
```