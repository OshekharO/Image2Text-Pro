import os
import tempfile
import numpy as np
import cv2
import pytest
import pytesseract
from program import TextExtractor

# Save original _validate_tesseract for testing
_original_validate_tesseract = TextExtractor._validate_tesseract
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


def test_validate_tesseract_error_handling(monkeypatch):
    def mock_get_version():
        raise Exception("Tesseract not found")

    monkeypatch.setattr(pytesseract, "get_tesseract_version", mock_get_version)
    extractor = TextExtractor.__new__(TextExtractor)
    with pytest.raises(RuntimeError) as exc_info:
        _original_validate_tesseract(extractor)
    assert "Tesseract OCR is not properly installed or accessible" in str(exc_info.value)


def test_extract_text_tesseract_error(monkeypatch, tmp_path):
    extractor = TextExtractor()
    img_path = str(tmp_path / "test.jpg")
    img = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.imwrite(img_path, img)

    def mock_image_to_string(*args, **kwargs):
        raise pytesseract.TesseractError(1, "Tesseract failed")

    monkeypatch.setattr(pytesseract, "image_to_string", mock_image_to_string)
    result = extractor.extract_text(img_path)
    assert result is None


def test_extract_text_general_exception(monkeypatch, tmp_path):
    extractor = TextExtractor()
    img_path = str(tmp_path / "test.jpg")
    img = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.imwrite(img_path, img)

    def mock_image_to_string(*args, **kwargs):
        raise Exception("Unexpected error")

    monkeypatch.setattr(pytesseract, "image_to_string", mock_image_to_string)
    result = extractor.extract_text(img_path)
    assert result is None


def test_process_images_invalid_directory():
    extractor = TextExtractor()
    stats = extractor.process_images("/path/does/not/exist/12345", "/tmp/out")
    assert stats == {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    with tempfile.NamedTemporaryFile() as tf:
        stats_file = extractor.process_images(tf.name, "/tmp/out")
        assert stats_file == {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}


def test_process_images_empty_directory():
    extractor = TextExtractor()
    with tempfile.TemporaryDirectory() as empty_dir, tempfile.TemporaryDirectory() as out_dir:
        stats = extractor.process_images(empty_dir, out_dir)
        assert stats == {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}


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
