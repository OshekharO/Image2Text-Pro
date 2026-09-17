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

def test_process_images_skip_existing(monkeypatch):
    extractor = TextExtractor()
    # Mock extract_text to return dummy text without executing pytesseract
    monkeypatch.setattr(extractor, 'extract_text', lambda img_path: "Extracted test text")

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create dummy image files
        img1 = os.path.join(input_dir, "page1.jpg")
        img2 = os.path.join(input_dir, "page2.jpg")
        open(img1, 'w').close()
        open(img2, 'w').close()

        # Pre-create output for page1.jpg
        out1 = os.path.join(output_dir, "page1.txt")
        open(out1, 'w').close()

        # Test process_images with skip_existing=True
        stats = extractor.process_images(input_dir, output_dir, skip_existing=True)
        assert stats['total'] == 2
        assert stats['skipped'] == 1
        assert stats['success'] == 1
        assert stats['failed'] == 0

        # Verify page2.txt was generated
        out2 = os.path.join(output_dir, "page2.txt")
        assert os.path.exists(out2)
