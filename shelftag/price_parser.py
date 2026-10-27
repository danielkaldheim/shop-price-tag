"""Turn OCR output from a cropped price tag into structured price data.

The OCR engine (ML Kit on device, RapidOCR here) gives text elements with
bounding boxes. Price tags print the main price in the largest font, often
with the øre as a raised, smaller number ("22⁸⁰"), and a smaller unit price
next to "pr KG" / "pr L" / "pr STK". Box geometry is therefore more reliable
than the raw text order, so the parser works on boxes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNIT_RE = re.compile(r"\b(?:pr\.?|per|/)\s*(kg|hg|l|ltr|liter|stk|pk|m)\b", re.I)
_PRICE_RE = re.compile(r"^(\d{1,4})(?:\s*[.,:]\s*(\d{2}|-{1,2}))?$")
_EAN_RE = re.compile(r"(?<!\d)(\d{8}|\d{12,14})(?!\d)")


@dataclass(frozen=True)
class TextBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class ParsedTag:
    price: float | None = None
    price_certainty: int = 0
    price_unit: str | None = None
    unit_price: float | None = None
    unit: str | None = None
    gtins: list[str] = field(default_factory=list)
    text: str = ""


def _clean(text: str) -> str:
    # Common OCR confusions on price digits.
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    t = text.strip().replace(" ", "")
    return t.translate(table) if re.fullmatch(r"[\dOoIlSB.,:\-]+", t) else text.strip()


def _as_price(text: str) -> tuple[int, int | None] | None:
    # Drop stray letters glued to the digits ("153.00k"), but not words
    # after a space, so "228 pr KG" is not mistaken for a price.
    m = _PRICE_RE.match(re.sub(r"(?<=[\d\-])[A-Za-z]{1,2}$", "", _clean(text)))
    if not m:
        return None
    kr = int(m.group(1))
    ore = m.group(2)
    if ore is None or "-" in ore:
        return kr, (0 if ore else None)
    return kr, int(ore)


def gtin_is_valid(code: str) -> bool:
    """Check the GS1 check digit of an EAN-8/UPC-A/EAN-13/GTIN-14 code."""
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    body, check = digits[:-1], digits[-1]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def find_gtins(text: str) -> list[str]:
    """GTINs printed as digits under the barcode, kept only if the check digit holds."""
    compact = re.sub(r"(?<=\d)[ ](?=\d)", "", text)
    seen: list[str] = []
    for code in _EAN_RE.findall(compact):
        if gtin_is_valid(code) and code not in seen:
            seen.append(code)
    return seen


def _main_price(boxes: list[TextBox]) -> tuple[float, TextBox, int] | None:
    """Return (price, its box, certainty).

    Certainty ranks how the øre were found, so readings of the same tag from
    different crops can be compared: 3 explicit (separator, ",-" or raised
    øre box), 2 split from merged digits, 1 bare whole number, 0 suspect.
    """
    candidates = [(b, p) for b in boxes if (p := _as_price(b.text)) is not None]
    if not candidates:
        return None
    main, (kr, ore) = max(candidates, key=lambda c: c[0].height)
    certainty = 3 if ore is not None else 1

    if ore is None:
        # Look for raised øre: a 2-digit box right of the main number, smaller,
        # with its top near the main number's top.
        best = None
        for b, _ in candidates:
            t = _clean(b.text)
            if b is main or not re.fullmatch(r"\d{2}", t):
                continue
            # OCR boxes around big glyphs are loose, so they may overlap.
            right = (b.x0 + b.x1) / 2 >= main.x1 - 0.1 * main.height and b.x0 <= main.x1 + 1.5 * main.height
            raised = abs(b.y0 - main.y0) <= 0.35 * main.height
            smaller = b.height <= 0.8 * main.height
            if right and raised and smaller and (best is None or b.x0 < best.x0):
                best = b
        if best is not None:
            ore, certainty = int(_clean(best.text)), 3
        elif kr >= 1000:
            # OCR merged "22⁸⁰" into "2280". Grocery prices of 1000 kr and up
            # are rare enough that splitting is the better bet.
            kr, ore = divmod(kr, 100)
            certainty = 2
    if _clean(main.text).startswith("0") and kr > 0:
        certainty = 0
    return kr + (ore or 0) / 100, main, certainty


def _units(
    boxes: list[TextBox], main: TextBox | None
) -> tuple[str | None, float | None, str | None]:
    """Return (unit of the main price, comparison price, its unit).

    "Pr/STK" printed beside the big number is the unit of the main price.
    "228 pr KG" elsewhere on the tag is the comparison (unit) price.
    """
    price_unit = unit_price = unit = None
    for b in boxes:
        m = _UNIT_RE.search(b.text)
        if not m:
            continue
        u = {"ltr": "l", "liter": "l"}.get(m.group(1).lower(), m.group(1).lower())
        nums = re.findall(r"\d+(?:[.,]\d{1,2})?", b.text[: m.start()])
        beside_main = (
            main is not None
            and b.x0 >= main.x0
            and main.y0 - 0.2 * main.height <= b.cy <= main.y1 + 0.3 * main.height
        )
        if nums and unit_price is None:
            unit_price, unit = float(nums[-1].replace(",", ".")), u
        elif beside_main and price_unit is None:
            price_unit = u
        elif unit is None:
            left = [
                o for o in boxes
                if o is not b and o is not main and _as_price(o.text)
                and abs(o.cy - b.cy) <= max(o.height, b.height) and o.x1 <= b.x0 + 2
            ]
            below = [
                o for o in boxes
                if o is not b and o is not main and _as_price(o.text)
                and 0 <= o.y0 - b.y0 <= 2 * b.height and o.x0 < b.x1 and o.x1 > b.x0
            ]
            if left or below:
                o = max(left, key=lambda o: o.x1) if left else min(below, key=lambda o: o.y0)
                kr, ore = _as_price(o.text)
                unit_price, unit = kr + (ore or 0) / 100, u
    return price_unit, unit_price, unit


def parse_tag(boxes: list[TextBox]) -> ParsedTag:
    boxes = [b for b in boxes if b.text.strip()]
    text = "\n".join(b.text for b in sorted(boxes, key=lambda b: (round(b.cy / 10), b.x0)))
    main = _main_price(boxes)
    price_unit, unit_price, unit = _units(boxes, main[1] if main else None)
    return ParsedTag(
        price=round(main[0], 2) if main else None,
        price_certainty=main[2] if main else 0,
        price_unit=price_unit,
        unit_price=unit_price,
        unit=unit,
        gtins=find_gtins(text),
        text=text,
    )
