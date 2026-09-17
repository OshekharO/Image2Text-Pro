import os
import tempfile
import numpy as np
import cv2
import pytest
from program import TextExtractor

# Mock tesseract validation so tests can run without tesseract binary installed
TextExtractor._validate_tesseract = lambda self: None


def test_clean_image_2d_grayscale():
    extractor = TextExtractor(preprocess=True)
    img_2d = np.random.randint(0, 256, (600, 800), dtype=np.uint8)
    cleaned = extractor.clean_image(img_2d)
    assert cleaned is not None
    assert isinstance(cleaned, np.ndarray)
    assert len(cleaned.shape) == 2


def test_clean_image_3d_bgr():
    extractor = TextExtractor(preprocess=True)
    img_3d = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    cleaned = extractor.clean_image(img_3d)
    assert cleaned is not None
    assert isinstance(cleaned, np.ndarray)
    assert len(cleaned.shape) == 2


def test_clean_extracted_text():
    extractor = TextExtractor()
    raw_text = "  Header Line  \n\n\n  Paragraph 1  \n\n  Paragraph 2  \n"
    cleaned = extractor._clean_extracted_text(raw_text)
    assert "Header Line" in cleaned
    assert "Paragraph 1" in cleaned
    assert "Paragraph 2" in cleaned
    # Ensure max 1 consecutive blank line
    assert "\n\n\n" not in cleaned


def test_extract_text_file_not_found():
    extractor = TextExtractor()
    result = extractor.extract_text("non_existent_file.jpg")
    assert result is None
