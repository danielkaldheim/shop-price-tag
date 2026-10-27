"""Pre-label shelf photos with Grounding DINO, for review before training.

Grounding DINO is too large to run in the app, but it finds price tags,
barcodes and QR codes zero-shot. Running it offline over new photos gives
draft YOLO labels that only need correcting in Roboflow (or any YOLO
labelling tool) instead of drawing every box by hand.

    python scripts/autolabel.py "test/rema" --out datasets/autolabel
    python scripts/autolabel.py photos/ --out datasets/autolabel --preview

Output is a YOLO dataset: images/, labels/ and data.yaml.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Prompt phrase -> class name. Order defines the class ids.
CLASSES = {
    "price tag": "price_tag",
    "barcode": "barcode",
    "qr code": "qr_code",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def label_for(text: str) -> int | None:
    for i, phrase in enumerate(CLASSES):
        if phrase in text:
            return i
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images", type=Path)
    p.add_argument("--out", type=Path, default=Path("datasets/autolabel"))
    p.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--preview", action="store_true", help="also save images with boxes drawn")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    prompt = " ".join(f"{phrase}." for phrase in CLASSES)

    files = sorted(f for f in args.images.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images found in {args.images}")
    for sub in ("images", "labels") + (("preview",) if args.preview else ()):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    for n, file in enumerate(files, 1):
        image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        w, h = image.size
        lines = []
        draw = ImageDraw.Draw(image) if args.preview else None
        texts = result.get("text_labels", result["labels"])
        for box, text in zip(result["boxes"].tolist(), texts):
            cls = label_for(text)
            if cls is None:
                continue
            x0, y0, x1, y1 = box
            # Grounding DINO sometimes returns the whole shelf as a "price tag".
            if (x1 - x0) * (y1 - y0) > 0.6 * w * h:
                continue
            lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            if draw:
                draw.rectangle(box, outline=("red", "lime", "blue")[cls], width=4)
                draw.text((x0, y0 - 12), text, fill="yellow")

        stem = f"{file.parent.name}_{file.stem}".replace(" ", "_")
        image_out = args.out / "images" / f"{stem}.jpg"
        if draw:
            image.save(args.out / "preview" / f"{stem}.jpg", quality=85)
            ImageOps.exif_transpose(Image.open(file)).convert("RGB").save(image_out, quality=95)
        else:
            image.save(image_out, quality=95)
        (args.out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        print(f"[{n}/{len(files)}] {file}: {len(lines)} boxes")

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES.values()))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images\nval: images\nnames:\n{names}\n"
    )


if __name__ == "__main__":
    main()
