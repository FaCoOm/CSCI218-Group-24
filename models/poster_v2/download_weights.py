"""
Download POSTER_V2 pretrained weights from Google Drive.

These files are excluded from the submission zip (total ~356 MB) but are
required to re-run inference.  Run this script once before running
inference_wrapper.py.

Usage:
    python download_weights.py

All files are saved relative to this script's directory.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    {
        "name": "RAF-DB trained checkpoint",
        "id":   "1aVm_hmJyZ5E_0p25XTbm3X9ophsKqCxv",
        "dest": os.path.join(SCRIPT_DIR, "checkpoint", "rafdb_best.pth"),
        "size": "241 MB",
    },
    {
        "name": "IR-50 face recognition backbone",
        "id":   "17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt",
        "dest": os.path.join(SCRIPT_DIR, "pretrain", "ir50.pth"),
        "size": "122 MB",
    },
    {
        "name": "MobileFaceNet landmark detector",
        "id":   "1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX",
        "dest": os.path.join(SCRIPT_DIR, "pretrain", "mobilefacenet_model_best.pth.tar"),
        "size": "12 MB",
    },
]


def check_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download_file(file_info: dict) -> None:
    dest = file_info["dest"]

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"  [SKIP] Already exists ({size_mb:.0f} MB): {os.path.basename(dest)}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {file_info['name']} ({file_info['size']}) ...")

    import gdown
    url = f"https://drive.google.com/uc?id={file_info['id']}"
    gdown.download(url, dest, quiet=False)
    print(f"  Saved -> {dest}")


def main():
    print("=" * 60)
    print("  POSTER_V2 — Weight Downloader")
    print("  Source: official POSTER_V2 repository (Google Drive)")
    print("=" * 60)

    check_gdown()

    for f in FILES:
        download_file(f)

    print()
    print("All weights ready.  You can now run:")
    print("  python inference_wrapper.py --config config.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
