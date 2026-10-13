"""Run fine-tuned LibreFace RAF-DB checkpoint inference and export predictions/libreface.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import CLASS_NAMES, Config, NORM_MEAN, NORM_STD
from model import LibreFaceResNet


def to_relative_path(abs_path: str, split: str) -> str:
    p = Path(abs_path)
    return f"data/raf_db/processed/{split}/{p.parent.name}/{p.name}"


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label in range(1, 8):
        class_dir = split_dir / str(label)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in valid_ext:
                samples.append((img_path, label))
    return samples


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=resampling)
        arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def load_finetuned_model(checkpoint_path: Path, cfg: Config, device: torch.device) -> LibreFaceResNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to create checkpoints_rafdb/libreface_rafdb_best.pt."
        )
    model = LibreFaceResNet(num_labels=7, dropout=cfg.dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
    return model


@torch.inference_mode()
def run_inference(cfg: Config, split: str, checkpoint_path: Path) -> pd.DataFrame:
    split_dir = {"train": cfg.train_dir, "val": cfg.val_dir, "test": cfg.test_dir}[split]
    samples = list_images(split_dir)
    device = torch.device(cfg.device)
    model = load_finetuned_model(checkpoint_path, cfg, device)
    records = []

    for start in tqdm(range(0, len(samples), cfg.batch_size), desc=f"Fine-tuned LibreFace {split}"):
        batch = samples[start:start + cfg.batch_size]
        images = [load_image_tensor(img_path, cfg.image_size) for img_path, _ in batch]
        x = torch.stack(images).to(device, non_blocking=True)
        logits7, _ = model(x)
        probs7 = F.softmax(logits7, dim=1)
        pred_labels = probs7.argmax(dim=1).cpu().numpy() + 1
        probs7_np = probs7.cpu().numpy()
        for i, (img_path, true_label) in enumerate(batch):
            row = {"image_path": to_relative_path(str(img_path), split),
                   "true_label": int(true_label), "pred_label": int(pred_labels[i])}
            for j in range(7):
                row[f"prob_{j + 1}"] = float(probs7_np[i, j])
            records.append(row)

    columns = ["image_path", "true_label", "pred_label", "prob_1", "prob_2", "prob_3",
               "prob_4", "prob_5", "prob_6", "prob_7"]
    return pd.DataFrame(records, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.model_dir / "checkpoints_rafdb" / "libreface_rafdb_best.pt"
    out_path = Path(args.output) if args.output else cfg.predictions_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_inference(cfg, args.split, checkpoint_path)
    df.to_csv(out_path, index=False)

    acc = (df["true_label"] == df["pred_label"]).mean() * 100
    prob_sums = df[[f"prob_{i}" for i in range(1, 8)]].sum(axis=1)
    print(f"\nSaved {len(df)} predictions to: {out_path}")
    print(f"{args.split.capitalize()} accuracy: {acc:.2f}%")
    print(f"Probability sum range: {prob_sums.min():.6f} to {prob_sums.max():.6f}")
    print("\nPredicted label distribution:")
    print(df["pred_label"].value_counts().sort_index().rename(index=CLASS_NAMES))


if __name__ == "__main__":
    main()
