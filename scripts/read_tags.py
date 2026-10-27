"""Run the full reference pipeline on photos: detect, crop, OCR, parse, decode.

This mirrors what famn does on device (YOLO via ultralytics_yolo, then ML Kit
text recognition and barcode scanning on each crop), using RapidOCR and
zxing-cpp so it can run on a laptop over the test folders.

    python scripts/read_tags.py "test/rema" --weights models/best.pt
    python scripts/read_tags.py test --json results.json --crops crops/
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import zxingcpp
from rapidocr import RapidOCR
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelftag.price_parser import TextBox, parse_tag

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def ocr_boxes(engine: RapidOCR, crop: np.ndarray, height: int) -> list[TextBox]:
    """OCR a crop after scaling it to a fixed height.

    Text detectors are trained on text that fills a small part of the image.
    A tight crop of a 22 kr price is mostly one huge glyph, which the detector
    misses at full resolution, so shrink it and add a border first.
    """
    scale = height / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    crop = cv2.copyMakeBorder(crop, 32, 32, 32, 32, cv2.BORDER_REPLICATE)
    out = engine(crop)
    if out.boxes is None:
        return []
    boxes = []
    for quad, text in zip(out.boxes, out.txts):
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        boxes.append(TextBox(text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def decode_codes(crop: np.ndarray) -> list[dict]:
    return [
        {"format": str(r.format).split(".")[-1], "text": r.text}
        for r in zxingcpp.read_barcodes(crop)
    ]


def _inside(inner: list[int], outer: list[int], slack: float = 0.1) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    sx, sy = (ox1 - ox0) * slack, (oy1 - oy0) * slack
    return ix0 >= ox0 - sx and iy0 >= oy0 - sy and ix1 <= ox1 + sx and iy1 <= oy1 + sy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images", type=Path)
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--pad", type=float, default=0.04, help="crop padding as a fraction of box size")
    p.add_argument("--tag-height", type=int, default=320, help="OCR input height for tag crops")
    p.add_argument("--price-height", type=int, default=160, help="OCR input height for price crops")
    p.add_argument("--json", type=Path)
    p.add_argument("--crops", type=Path, help="save each crop here for inspection")
    args = p.parse_args()

    model = YOLO(args.weights)
    engine = RapidOCR()
    files = [args.images] if args.images.is_file() else sorted(
        f for f in args.images.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.crops:
        args.crops.mkdir(parents=True, exist_ok=True)

    report = []
    for file in files:
        image = cv2.imread(str(file))
        result = model(image, conf=args.conf, verbose=False)[0]
        h, w = image.shape[:2]
        dets = [
            (result.names[int(c)], [int(v) for v in xyxy], round(conf, 3))
            for xyxy, conf, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        # The v4 model has two classes: the whole tag ("price-tags") and the
        # big price on it ("price-tag-price").
        tag_boxes = [d for d in dets if not d[0].endswith("price")]
        price_boxes = [d for d in dets if d[0].endswith("price")]

        def crop(box):
            x0, y0, x1, y1 = box
            px, py = (x1 - x0) * args.pad, (y1 - y0) * args.pad
            x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
            x1, y1 = min(w, int(x1 + px)), min(h, int(y1 + py))
            return image[y0:y1, x0:x1]

        tags = []
        for i, (_, box, conf) in enumerate(tag_boxes):
            tag_crop = crop(box)
            parsed = parse_tag(ocr_boxes(engine, tag_crop, args.tag_height))
            # The v4 model also boxes the big price itself. OCR that tight crop
            # too and keep whichever reading found the øre more reliably;
            # ties go to the price crop, where the digits are largest.
            inner = [p for p in price_boxes if _inside(p[1], box)]
            price_source = "tag"
            if inner:
                best_inner = max(inner, key=lambda p: p[2])[1]
                price_only = parse_tag(ocr_boxes(engine, crop(best_inner), args.price_height))
                if price_only.price is not None and price_only.price_certainty >= parsed.price_certainty:
                    parsed.price = price_only.price
                    parsed.price_certainty = price_only.price_certainty
                    parsed.price_unit = parsed.price_unit or price_only.price_unit
                    price_source = "price_box"
            tags.append({
                "box": box,
                "confidence": conf,
                **asdict(parsed),
                "price_source": price_source,
                "barcodes": decode_codes(tag_crop),
            })
            if args.crops:
                cv2.imwrite(str(args.crops / f"{file.parent.name}_{file.stem}_{i}.jpg"), tag_crop)
        report.append({"image": str(file), "tags": tags})
        summary = "; ".join(
            f"{t['price']} kr/{t['price_unit'] or '?'}"
            + (f" ({t['unit_price']}/{t['unit']})" if t["unit_price"] else "")
            + f" gtin={t['gtins'] or [c['text'] for c in t['barcodes']]}"
            for t in tags
        ) or "no tags"
        print(f"{file}: {summary}", flush=True)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
