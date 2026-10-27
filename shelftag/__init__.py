"""Reference pipeline for reading Norwegian shelf price tags.

The app runs detection on-device through ultralytics_yolo and does OCR and
barcode decoding with ML Kit. This package mirrors that pipeline in Python so
it can be evaluated on the test photos and kept in sync with the Dart code.
"""
