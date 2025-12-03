#!/usr/bin/env python3
"""
Download MedicalNet pretrained weights.

MedicalNet provides 3D ResNet models pretrained on 23 medical imaging datasets.
Paper: https://arxiv.org/abs/1904.00625
GitHub: https://github.com/Tencent/MedicalNet

Usage:
    python download_medicalnet.py --depth 10
    python download_medicalnet.py --depth 18
    python download_medicalnet.py --all
"""

import argparse
import os
from pathlib import Path
import urllib.request
import hashlib

# MedicalNet pretrained weight URLs (from official repository)
MEDICALNET_URLS = {
    10: {
        "url": "https://drive.google.com/uc?export=download&id=1Rp0H7jvzq4TdB8FgLZ6nVmSSTdLB_6pv",
        "filename": "resnet_10_23dataset.pth",
        "description": "ResNet-10 (smallest, ~5M params) - good for limited data"
    },
    18: {
        "url": "https://drive.google.com/uc?export=download&id=1i6HG7LPOuL2P8S8jBqLZOjNqEKzzUFPJ",
        "filename": "resnet_18_23dataset.pth",
        "description": "ResNet-18 (~11M params) - balanced"
    },
    34: {
        "url": "https://drive.google.com/uc?export=download&id=1kG7l3n33p7Y3yl6E9gD8XQEM0Y3vCmVB",
        "filename": "resnet_34_23dataset.pth",
        "description": "ResNet-34 (~21M params) - more capacity"
    },
    50: {
        "url": "https://drive.google.com/uc?export=download&id=1WOXx2A8tVoI_h9rInhFXH3D0BNGj0l-I",
        "filename": "resnet_50_23dataset.pth",
        "description": "ResNet-50 (~46M params) - largest, needs more data"
    }
}

EXPERIMENT_DIR = Path(__file__).parent


def download_from_gdrive(file_id, destination):
    """Download file from Google Drive."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        return True
    except ImportError:
        print("gdown not installed. Installing...")
        os.system("pip install gdown")
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_gdrive_id(url):
    """Extract file ID from Google Drive URL."""
    if "id=" in url:
        return url.split("id=")[-1]
    elif "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    return url


def download_weights(depth, output_dir):
    """Download MedicalNet weights for specified depth."""
    if depth not in MEDICALNET_URLS:
        print(f"Error: Unsupported depth {depth}. Choose from: {list(MEDICALNET_URLS.keys())}")
        return False

    info = MEDICALNET_URLS[depth]
    output_path = output_dir / info["filename"]

    if output_path.exists():
        print(f"Weights already exist: {output_path}")
        return True

    print(f"\nDownloading ResNet-{depth} pretrained weights...")
    print(f"Description: {info['description']}")
    print(f"Destination: {output_path}")

    file_id = extract_gdrive_id(info["url"])
    success = download_from_gdrive(file_id, output_path)

    if success and output_path.exists():
        print(f"Successfully downloaded: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    else:
        print(f"Failed to download weights for ResNet-{depth}")
        print("\nManual download instructions:")
        print(f"1. Go to: https://github.com/Tencent/MedicalNet")
        print(f"2. Download {info['filename']} from the Google Drive links")
        print(f"3. Place it in: {output_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MedicalNet pretrained weights")
    parser.add_argument("--depth", type=int, choices=[10, 18, 34, 50],
                        help="ResNet depth to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all available weights")
    parser.add_argument("--output", type=str, default="pretrained",
                        help="Output directory (default: pretrained/)")
    args = parser.parse_args()

    output_dir = EXPERIMENT_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedicalNet Pretrained Weights Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")

    if args.all:
        depths = list(MEDICALNET_URLS.keys())
    elif args.depth:
        depths = [args.depth]
    else:
        # Default: download ResNet-10 (smallest, good for limited data)
        depths = [10]
        print("\nNo depth specified. Downloading ResNet-10 (recommended for limited data)")

    print("\nAvailable models:")
    for d, info in MEDICALNET_URLS.items():
        status = "downloading" if d in depths else "skipped"
        print(f"  ResNet-{d}: {info['description']} [{status}]")

    for depth in depths:
        download_weights(depth, output_dir)

    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("\nTo use pretrained weights, update config.yaml:")
    print(f'  pretrained_path: "pretrained/{MEDICALNET_URLS[depths[0]]["filename"]}"')
    print("\nRecommended training strategy:")
    print("  1. First, freeze encoder and train only LSTM + classifier:")
    print("     freeze_encoder: true")
    print("  2. Then, unfreeze and fine-tune the whole network:")
    print("     freeze_encoder: false")
    print("     learning_rate: 0.00001  # Lower LR for fine-tuning")


if __name__ == "__main__":
    main()
