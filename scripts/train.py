"""Train the on-device price tag detector.

Downloads a dataset version from Roboflow (ROBOFLOW_API_KEY in .env) or uses a
local data.yaml, then fine-tunes a YOLO26 nano model. Nano is the size the
ultralytics_yolo plugin ships for real-time camera use.

    python scripts/train.py --version 4
    python scripts/train.py --data datasets/autolabel/data.yaml --model yolo26s.pt
"""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = "daniel-kaldheim-4f58q"
PROJECT = "price-tags-c79pp"


def download(version: int) -> Path:
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY") or dotenv_values(ROOT / ".env")["ROBOFLOW_API_KEY"]
    project = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(version).download("yolov11", location=str(ROOT / "datasets" / f"price-tags-{version}"))
    return Path(dataset.location) / "data.yaml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, default=4, help="Roboflow dataset version")
    p.add_argument("--data", type=Path, help="local data.yaml, skips the Roboflow download")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cuda index, mps or cpu; auto when omitted")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    data = args.data or download(args.version)
    model = YOLO(args.model)
    model.train(
        data=str(data),
        project=str(ROOT / "runs"),
        name=args.name or f"{Path(args.model).stem}-price-tag",
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=-1,
        patience=50,
        # Tags are photographed at an angle and in poor light, but never
        # mirrored: flipping would teach the model backwards digits.
        fliplr=0.0,
        degrees=10.0,
        perspective=0.0005,
        hsv_v=0.5,
    )


if __name__ == "__main__":
    main()
