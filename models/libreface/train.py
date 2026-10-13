"""Fine-tune LibreFace ResNet on RAF-DB.

Starts from the pretrained LibreFace AffectNet checkpoint, replaces the final
8-class layer with a 7-class RAF-DB layer, and saves the best validation model.
Project labels: 1 Surprise, 2 Fear, 3 Disgust, 4 Happiness, 5 Sadness, 6 Anger, 7 Neutral.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import Config, CLASS_NAMES, NORM_MEAN, NORM_STD
from model import LibreFaceResNet


class RAFDBFolderDataset(Dataset):
    """Read RAF-DB images from folders 1..7 and convert labels to 0..6 for training."""

    def __init__(self, split_dir: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for label in range(1, 8):
            class_dir = split_dir / str(label)
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_ext:
                    self.samples.append((img_path, label - 1))

        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return x, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train_tf, eval_tf


def compute_class_weights(dataset: RAFDBFolderDataset, device: torch.device) -> torch.Tensor:
    labels = np.array([label for _, label in dataset.samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=7)
    weights = counts.sum() / (7 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_affectnet_encoder_weights(model: LibreFaceResNet, checkpoint_path: Path, device: torch.device) -> None:
    """Load compatible weights from 8-class AffectNet checkpoint and skip the final 8-class layer."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            "Expected: models/libreface/checkpoints_fm_resnet/AffectNet/resnet.pt"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model_state = model.state_dict()
    compatible_state: dict[str, Any] = {}
    skipped: list[str] = []

    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)

    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=True)

    print(f"Loaded compatible pretrained tensors: {len(compatible_state)}")
    print(f"Skipped incompatible tensors: {skipped}")


def set_encoder_trainable(model: LibreFaceResNet, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)


def run_one_epoch(
    model: LibreFaceResNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for x, y in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))

    return total_loss / total, correct / total


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_ds = RAFDBFolderDataset(cfg.train_dir, train_tf)
    val_ds = RAFDBFolderDataset(cfg.val_dir, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    model = LibreFaceResNet(num_labels=7, dropout=cfg.dropout).to(device)
    pretrained_path = Path(args.pretrained) if args.pretrained else cfg.checkpoint_file
    load_affectnet_encoder_weights(model, pretrained_path, device)

    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, False)
        print(f"Encoder frozen for first {args.freeze_encoder_epochs} epoch(s).")

    class_weight_tensor = compute_class_weights(train_ds, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.05)

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    out_path = Path(args.output) if args.output else (cfg.model_dir / "checkpoints_rafdb" / "libreface_rafdb_best.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_encoder_epochs + 1 and args.freeze_encoder_epochs > 0:
            set_encoder_trainable(model, True)
            optimizer = make_optimizer(model, args.lr * 0.5, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            print("Encoder unfrozen for fine-tuning.")

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)

        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f}, train acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f}, val acc {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                        "val_loss": val_loss, "class_names": CLASS_NAMES,
                        "source": "LibreFace AffectNet checkpoint fine-tuned on RAF-DB"}, out_path)
            print(f"Saved best checkpoint: {out_path}")
        else:
            bad_epochs += 1

        save_history(history, cfg.log_dir / "libreface_rafdb_train_history.json")

        if bad_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epoch(s) without improvement.")
            break

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
