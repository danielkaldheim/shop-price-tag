"""Export trained weights for the ultralytics_yolo Flutter plugin.

Android gets a LiteRT (.tflite) model and iOS a Core ML .mlpackage, zipped so
it can be shipped as a Flutter asset under assets/models/. Both use a 640x640
input, matching the plugin's own assets.

    python scripts/export.py runs/yolo26n-price-tag/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=Path("models/export"))
    p.add_argument("--platform", choices=["android", "ios", "both"], default="both")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.weights.parent.parent.name if args.weights.parent.name == "weights" else args.weights.stem

    if args.platform in ("android", "both"):
        # nms=None keeps raw outputs and lets the plugin run NMS on device.
        # w8a32 (INT8 weights, FP32 activations) matches the plugin's own
        # Android assets and needs no calibration data.
        path = Path(YOLO(args.weights).export(format="litert", nms=None, quantize="w8a32", imgsz=args.imgsz))
        target = args.out / f"{name}.tflite"
        shutil.copy(next(path.rglob("*.tflite")) if path.is_dir() else path, target)
        print("android:", target)

    if args.platform in ("ios", "both"):
        path = Path(YOLO(args.weights).export(format="coreml", nms=True, quantize=8, imgsz=args.imgsz))
        target = args.out / f"{name}.mlpackage"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(path, target)
        archive = shutil.make_archive(str(target), "zip", root_dir=args.out, base_dir=target.name)
        print("ios:", archive)


if __name__ == "__main__":
    main()
