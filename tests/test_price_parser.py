from shelftag.price_parser import TextBox, find_gtins, gtin_is_valid, parse_tag


def box(text, x0, y0, x1, y1):
    return TextBox(text, x0, y0, x1, y1)


def test_raised_ore_and_unit_price():
    # Layout of test/rema/IMG_4290.jpeg (crop coordinates).
    tag = parse_tag([
        box("SMOOTHIE QU/GULROT", 20, 20, 380, 50),
        box("KOLONIHAGEN 100 G", 20, 60, 180, 75),
        box("7032069733448", 210, 80, 350, 95),
        box("A1", 20, 120, 50, 140),
        box("22", 200, 100, 300, 170),
        box("80", 315, 100, 360, 130),
        box("Pr/STK", 310, 145, 370, 165),
        box("228 pr KG", 20, 150, 160, 175),
    ])
    assert tag.price == 22.80
    assert tag.price_unit == "stk"
    assert (tag.unit_price, tag.unit) == (228.0, "kg")
    assert tag.gtins == ["7032069733448"]


def test_price_per_kg_without_comparison_price():
    # Layout of test/norges gruppen/IMG_3930.jpeg.
    tag = parse_tag([
        box("LIME", 60, 10, 120, 30),
        box("BRASIL", 60, 32, 110, 44),
        box("BAMA-GRUPPEN AS", 60, 46, 170, 58),
        box("59", 180, 30, 240, 80),
        box("90", 245, 30, 275, 50),
        box("Pr.KG", 245, 58, 285, 74),
    ])
    assert tag.price == 59.90
    assert tag.price_unit == "kg"
    assert tag.unit_price is None


def test_merged_ore_is_split():
    assert parse_tag([box("2280", 0, 0, 200, 80)]).price == 22.80


def test_separators_and_dash():
    assert parse_tag([box("39,90", 0, 0, 100, 50)]).price == 39.90
    assert parse_tag([box("49,-", 0, 0, 100, 50)]).price == 49.0
    assert parse_tag([box("149", 0, 0, 100, 50)]).price == 149.0


def test_trailing_ocr_noise():
    assert parse_tag([box("153.00k", 0, 0, 100, 50)]).price == 153.0
    assert parse_tag([box("228 pr KG", 0, 0, 100, 50)]).price is None


def test_ocr_letter_confusion():
    assert parse_tag([box("2O", 0, 0, 80, 60), box("5O", 85, 0, 110, 25)]).price == 20.50


def test_gtin_checksum():
    assert gtin_is_valid("7032069733448")
    assert not gtin_is_valid("7032069733449")
    assert find_gtins("ean 7032069733448 and 7032069733449") == ["7032069733448"]
    assert find_gtins("7032 0697 3344 8") == ["7032069733448"]


def test_unit_price_below_label():
    # Coop prints the comparison price under its "Pr/KG" label.
    tag = parse_tag([
        box("22", 385, 98, 640, 312),
        box("70", 578, 120, 698, 203),
        box("Pr/Stk", 591, 210, 699, 242),
        box("Pr/KG", 110, 264, 215, 295),
        box("50.44", 108, 289, 289, 335),
    ])
    assert (tag.price, tag.price_unit) == (22.70, "stk")
    assert (tag.unit_price, tag.unit) == (50.44, "kg")


def test_certainty():
    assert parse_tag([box("22", 0, 0, 100, 80), box("80", 105, 0, 140, 40)]).price_certainty == 3
    assert parse_tag([box("2280", 0, 0, 200, 80)]).price_certainty == 2
    assert parse_tag([box("90", 0, 0, 100, 80)]).price_certainty == 1
    assert parse_tag([box("0669", 0, 0, 100, 80)]).price_certainty == 0
