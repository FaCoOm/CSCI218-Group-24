"""Configuration for the LibreFace FER baseline wrapper.

This wrapper uses the uploaded LibreFace AffectNet ResNet checkpoint directly
and exports predictions/libreface.csv
"""

# Author: Quoc Minh Luong
# Created on: 11 October 2026

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch


# Project label mapping. This must match eval/utils.py and predictions/README.md.
CLASS_NAMES: dict[int, str] = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

# LibreFace AffectNet checkpoint output order.
# The pretrained ResNet has 8 outputs: Neutral, Happiness, Sadness,
# Surprise, Fear, Disgust, Anger, Contempt.
AFFECTNET_CLASS_NAMES: dict[int, str] = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}

# Convert 8-class AffectNet output to this project's 7-class RAF-DB mapping.
# Contempt is not present in the project labels, so it is folded into Neutral.
AFFECTNET_TO_PROJECT_LABEL: dict[int, int] = {
    0: 7,  # Neutral  -> Neutral
    1: 4,  # Happiness -> Happiness
    2: 5,  # Sadness -> Sadness
    3: 1,  # Surprise -> Surprise
    4: 2,  # Fear -> Fear
    5: 3,  # Disgust -> Disgust
    6: 6,  # Anger -> Anger
    7: 7,  # Contempt -> Neutral
}

# ImageNet statistics used by the uploaded LibreFace data.py.
NORM_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORM_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class Config:
    # Image / data
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    num_classes_project: int = 7
    num_classes_libreface: int = 8

    # Model settings copied from the LibreFace checkpoint structure
    dropout: float = 0.1
    hidden_dim: int = 128
    student_model_name: str = "resnet" 
    data: str = "AffectNet"

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Derived paths
    project_root: Path = field(init=False)
    data_root: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)

    model_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    checkpoint_file: Path = field(init=False)
    predictions_csv: Path = field(init=False)
    debug_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        # config.py lives at: <root>/models/libreface/config.py
        self.project_root = Path(__file__).resolve().parents[2]

        self.data_root = self.project_root / "data" / "raf_db" / "processed"
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"

        self.model_dir = self.project_root / "models" / "libreface"
        self.checkpoint_dir = self.model_dir / "checkpoints_fm_resnet" / "AffectNet"
        self.log_dir = self.model_dir / "logs"
        self.predictions_dir = self.project_root / "predictions"

        self.checkpoint_file = self.checkpoint_dir / "resnet.pt"
        self.predictions_csv = self.predictions_dir / "libreface.csv"
        self.debug_csv = self.log_dir / "libreface_debug.csv"

        for d in (self.checkpoint_dir, self.log_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)
