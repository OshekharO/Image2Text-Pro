# Forge Journal

## 2026-05-18 - Baseline Verification & Test Suite Expansion

**Learning:** `TextExtractor._validate_tesseract` is executed during `__init__`. When testing `TextExtractor` without a local Tesseract installation, monkeypatching `_validate_tesseract` or handling `pytesseract.TesseractError` / `RuntimeError` is essential for clean test execution across environments.
**Action:** Mock `get_tesseract_version` or `_validate_tesseract` appropriately in unit tests to test both normal initialization and error paths cleanly.
