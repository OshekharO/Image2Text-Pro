import os
import cv2
import numpy as np
import pytesseract
from typing import Optional, List, Tuple, Dict
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import platform
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def configure_tesseract_path() -> bool:
    """
    Auto-configure Tesseract executable path for Windows systems.
    
    Returns:
        True if Tesseract was found and configured, False otherwise
    """
    if platform.system() != 'Windows':
        return True  # On Linux/Mac, rely on PATH
    
    # Check if already configured
    current_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', 'tesseract')
    if current_cmd != 'tesseract' and os.path.exists(current_cmd):
        return True
    
    # Common installation paths on Windows
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe'),
        os.path.expandvars(r'%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
    ]
    
    # Try common paths
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Auto-configured Tesseract path: {path}")
            return True
    
    # Check if tesseract is in PATH
    tesseract_in_path = shutil.which('tesseract')
    if tesseract_in_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_in_path
        logger.info(f"Found Tesseract in PATH: {tesseract_in_path}")
        return True
    
    return False


class TextExtractor:
    """Text extraction from images using Tesseract OCR with preprocessing."""
    
    # Preprocessing constants for optimal OCR performance
    MIN_HEIGHT_FOR_UPSCALE = 500  # Images smaller than this will be upscaled
    UPSCALE_FACTOR = 2  # Factor to scale small images
    BLOCK_SIZE_UPSCALED = 15  # Adaptive threshold block size for upscaled images
    BLOCK_SIZE_NORMAL = 11  # Adaptive threshold block size for normal images
    MIN_TEXT_LENGTH = 1  # Minimum text length to save
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    def __init__(
        self, 
        lang: str = 'chi_sim', 
        psm: int = 6, 
        preprocess: bool = True,
        tesseract_cmd: Optional[str] = None
    ):
        """
        Initialize the TextExtractor with configuration options.
        
        Args:
            lang: Language for OCR (default: 'chi_sim' for simplified Chinese)
            psm: Page segmentation mode (default: 6 for assuming uniform block of text)
            preprocess: Whether to apply image preprocessing (default: True)
            tesseract_cmd: Path to tesseract executable (optional)
        """
        self.lang = lang
        self.psm = psm
        self.preprocess = preprocess
        
        # Configure Tesseract path if explicitly provided
        if tesseract_cmd:
            if not os.path.exists(tesseract_cmd):
                raise FileNotFoundError(f"Tesseract executable not found: {tesseract_cmd}")
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Validate Tesseract is available
        self._validate_tesseract()
    
    def _validate_tesseract(self) -> None:
        """Validate that Tesseract is installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            raise RuntimeError(
                f"Tesseract OCR is not properly installed or accessible: {e}\n"
                "Installation guide:\n"
                "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  - Linux: sudo apt install tesseract-ocr tesseract-ocr-chi-sim\n"
                "  - macOS: brew install tesseract tesseract-lang"
            )
    
    def clean_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Clean the image to improve OCR accuracy.
        Optimized for novel text extraction with both small and large fonts.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image or None if processing fails
        """
        try:
            # Convert to grayscale if needed
            # Performance optimization: Avoid redundant array copy when image is already 2D grayscale,
            # since gray is not mutated in place by downstream OpenCV calls.
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Get image dimensions to adapt processing
            height, width = gray.shape
            
            # Upscale small images to improve OCR on small fonts
            # Tesseract works best with text height of 20-30 pixels
            scale_factor = 1
            if height < self.MIN_HEIGHT_FOR_UPSCALE:
                scale_factor = self.UPSCALE_FACTOR
                gray = cv2.resize(
                    gray, None, fx=scale_factor, fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Light denoising - reduced strength to preserve small text details
            denoised = cv2.fastNlMeansDenoising(
                gray, None, h=5,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            # Adaptive thresholding with larger block size for better text preservation
            # Block size must be odd and > 1
            block_size = self.BLOCK_SIZE_UPSCALED if scale_factor > 1 else self.BLOCK_SIZE_NORMAL
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, block_size)  # Ensure minimum valid size
            
            thresh = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive blank lines and whitespace.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Split into lines and strip each
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove excessive blank lines (keep maximum one consecutive blank line)
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append(line)
                prev_blank = True
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_text(self, image_path: str) -> Optional[str]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text or None if extraction fails or no text found
        """
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None
            
            # Read image directly as grayscale for performance and memory efficiency.
            # Performance impact: IMREAD_GRAYSCALE instructs C/C++ image decoders to decode 1-channel grayscale directly,
            # reducing image load time and peak image memory allocation by ~50%.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Unable to read image (may be corrupt or unsupported): {image_path}")
                return None
            
            # Preprocess or just convert to grayscale
            if self.preprocess:
                processed_image = self.clean_image(image)
                if processed_image is None:
                    return None
            else:
                # Handle fallback if a 3-channel image array is passed
                if len(image.shape) == 3:
                    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    processed_image = image
            
            # Build OCR configuration
            config = f'--psm {self.psm} --oem 3'  # OEM 3 = default OCR engine mode
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config=config
            )
            
            # Clean the extracted text
            text = self._clean_extracted_text(text)
            
            return text if text and len(text) >= self.MIN_TEXT_LENGTH else None
            
        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract error for {image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Text extraction failed for {image_path}: {e}")
            return None
    
    def process_images(
        self,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        batch_size: int = 10,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save text files
            max_workers: Maximum number of parallel workers
            batch_size: Number of images to process before logging progress
            skip_existing: Skip files that already have output text files
            
        Returns:
            Dictionary with processing statistics:
            - total: Total images found
            - success: Successfully processed
            - failed: Failed to extract text
            - skipped: Skipped (already existed)
        """
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Validate input directory
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return stats
        
        if not os.path.isdir(input_dir):
            logger.error(f"Input path is not a directory: {input_dir}")
            return stats
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files and SORT them for consistent ordering (critical for novels)
        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ])
        
        if not image_files:
            logger.warning(f"No supported images found in {input_dir}")
            logger.info(f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}")
            return stats
        
        stats['total'] = len(image_files)
        logger.info(f"Found {len(image_files)} images to process")
        
        # Thread-safe progress tracking
        start_time = time.time()
        lock = threading.Lock()
        processed_count = 0
        
        def process_single_file(image_file: str) -> Tuple[str, bool, str]:
            """
            Process a single image file.
            
            Returns:
                Tuple of (filename, success, status_message)
            """
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(image_file)[0]}.txt"
            )
            
            # Skip if output already exists and skip_existing is enabled
            if skip_existing and os.path.exists(output_path):
                return (image_file, True, 'skipped')
            
            # Extract text
            text = self.extract_text(image_path)
            
            if text and len(text) >= self.MIN_TEXT_LENGTH:
                try:
                    with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                        f.write(text)
                        f.write('\n')  # Ensure file ends with newline
                    return (image_file, True, 'success')
                except IOError as e:
                    return (image_file, False, f'write error: {e}')
            else:
                return (image_file, False, 'no text extracted')
        
        # Performance optimization: Pre-filter skipped files on the main thread when skip_existing is True.
        # Avoids generating Futures and dispatching worker threads for files whose output already exists.
        files_to_process = []
        if skip_existing:
            for img in image_files:
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(img)[0]}.txt"
                )
                if os.path.exists(output_path):
                    stats['skipped'] += 1
                    processed_count += 1
                else:
                    files_to_process.append(img)
        else:
            files_to_process = image_files

        if processed_count > 0:
            logger.info(f"Skipped {stats['skipped']} already processed files upfront")

        # Process remaining images in parallel using ThreadPoolExecutor
        if files_to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit only unskipped tasks
                future_to_file = {
                    executor.submit(process_single_file, img): img
                    for img in files_to_process
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    image_file, success, message = future.result()
                    
                    # Thread-safe update of stats and progress
                    with lock:
                        processed_count += 1

                        if message == 'skipped':
                            stats['skipped'] += 1
                        elif success:
                            stats['success'] += 1
                        else:
                            stats['failed'] += 1
                            logger.warning(f"Failed: {image_file} - {message}")

                        # Log progress at batch intervals or at completion
                        if processed_count % batch_size == 0 or processed_count == len(image_files):
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"Progress: {processed_count}/{len(image_files)} "
                                f"({rate:.2f} img/s) | "
                                f"OK: {stats['success']}, "
                                f"Fail: {stats['failed']}, "
                                f"Skip: {stats['skipped']}"
                            )
        
        # Final summary
        total_time = time.time() - start_time
        avg_time = total_time / len(image_files) if image_files else 0
        logger.info(
            f"\n{'='*50}\n"
            f"Processing Complete\n"
            f"{'='*50}\n"
            f"Total time: {total_time:.2f}s\n"
            f"Average per image: {avg_time*1000:.1f}ms\n"
            f"Results: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['skipped']} skipped\n"
            f"{'='*50}"
        )
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from images using Tesseract OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i ./images -o ./output
  %(prog)s -i ./images -o ./output -l eng --psm 3
  %(prog)s -i ./images -o ./output --workers 8 --no-preprocess
  %(prog)s -i ./images -o ./output -l "chi_sim+eng" --no-skip

PSM Modes (Page Segmentation):
  0    = Orientation and script detection (OSD) only
  1    = Automatic page segmentation with OSD
  3    = Fully automatic page segmentation (no OSD, good for mixed content)
  6    = Assume a single uniform block of text (default, good for novels)
  11   = Sparse text (find as much text as possible)
  13   = Raw line by line recognition

Common Languages:
  chi_sim  = Simplified Chinese
  chi_tra  = Traditional Chinese  
  eng      = English
  jpn      = Japanese
  kor      = Korean
  chi_sim+eng = Chinese + English (combined)
        """
    )
    
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Output directory for text files'
    )
    parser.add_argument(
        '-l', '--lang', default='chi_sim',
        help='OCR language (default: chi_sim). Use "chi_sim+eng" for mixed content.'
    )
    parser.add_argument(
        '--psm', type=int, default=6,
        help='Page segmentation mode (default: 6)'
    )
    parser.add_argument(
        '--no-preprocess', action='store_false', dest='preprocess',
        help='Disable image preprocessing (faster but less accurate)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Images between progress updates (default: 10)'
    )
    parser.add_argument(
        '--tesseract-cmd', type=str, default=None,
        help='Path to tesseract executable (auto-detected on Windows)'
    )
    parser.add_argument(
        '--no-skip', action='store_false', dest='skip_existing',
        help='Reprocess images even if output already exists'
    )
    
    args = parser.parse_args()
    
    # Auto-configure Tesseract path on Windows if not explicitly provided
    if platform.system() == 'Windows' and args.tesseract_cmd is None:
        if not configure_tesseract_path():
            logger.warning(
                "Could not auto-detect Tesseract path on Windows.\n"
                "Please specify with --tesseract-cmd or install Tesseract.\n"
                "Download: https://github.com/UB-Mannheim/tesseract/wiki"
            )
    
    try:
        # Initialize extractor (validates Tesseract installation)
        extractor = TextExtractor(
            lang=args.lang,
            psm=args.psm,
            preprocess=args.preprocess,
            tesseract_cmd=args.tesseract_cmd
        )
        
        # Process images
        stats = extractor.process_images(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.workers,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing
        )
        
        # Exit with error code if all processing failed
        if stats['total'] > 0 and stats['success'] == 0 and stats['skipped'] == 0:
            logger.error("All images failed to process!")
            exit(1)
        
    except RuntimeError as e:
        logger.error(str(e))
        exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user (Ctrl+C)")
        exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
