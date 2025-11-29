import os
import cv2
import numpy as np
import pytesseract
from typing import Optional, List
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextExtractor:
    # Preprocessing constants for optimal OCR performance
    MIN_HEIGHT_FOR_UPSCALE = 500  # Images smaller than this will be upscaled
    UPSCALE_FACTOR = 2  # Factor to scale small images
    BLOCK_SIZE_UPSCALED = 15  # Adaptive threshold block size for upscaled images
    BLOCK_SIZE_NORMAL = 11  # Adaptive threshold block size for normal images
    
    def __init__(self, lang: str = 'chi_sim', psm: int = 6, preprocess: bool = True):
        """
        Initialize the TextExtractor with configuration options.
        
        Args:
            lang: Language for OCR (default: 'chi_sim' for simplified Chinese)
            psm: Page segmentation mode (default: 6 for assuming uniform block of text)
            preprocess: Whether to apply image preprocessing (default: True)
        """
        self.lang = lang
        self.psm = psm
        self.preprocess = preprocess
        
    def clean_image(self, image) -> Optional[cv2.Mat]:
        """
        Clean the image to improve OCR accuracy.
        Optimized for novel text extraction with both small and large fonts.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image or None if processing fails
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get image dimensions to adapt processing
            height, width = gray.shape
            
            # Upscale small images to improve OCR on small fonts
            # Tesseract works best with text height of 20-30 pixels
            scale_factor = 1
            if height < self.MIN_HEIGHT_FOR_UPSCALE:
                scale_factor = self.UPSCALE_FACTOR
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                                 interpolation=cv2.INTER_CUBIC)
            
            # Light denoising - reduced strength to preserve small text details
            denoised = cv2.fastNlMeansDenoising(gray, None, h=5, 
                                                templateWindowSize=7, 
                                                searchWindowSize=21)
            
            # Adaptive thresholding with larger block size for better text preservation
            # Block size should be odd and larger for varied text sizes
            block_size = self.BLOCK_SIZE_UPSCALED if scale_factor > 1 else self.BLOCK_SIZE_NORMAL
            thresh = cv2.adaptiveThreshold(denoised, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, block_size, 2)
            
            return thresh
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_text(self, image_path: str) -> Optional[str]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Unable to read image: {image_path}")
                return None
                
            # Preprocess if enabled
            processed_image = self.clean_image(image) if self.preprocess else image
            
            # OCR configuration
            config = f'--psm {self.psm} --oem 3'  # oem 3 = default OCR engine
            if self.lang:
                config += f' -l {self.lang}'
                
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image, 
                config=config
            )
            
            return text.strip()
        except Exception as e:
            logger.error(f"Text extraction failed for {image_path}: {e}")
            return None
    
    def process_images(
        self, 
        input_dir: str, 
        output_dir: str, 
        max_workers: int = 4,
        batch_size: int = 10
    ) -> None:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save text files
            max_workers: Maximum number of parallel workers
            batch_size: Number of images to process before logging progress
        """
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))
        ]
        
        if not image_files:
            logger.warning(f"No supported images found in {input_dir}")
            return
            
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        start_time = time.time()
        processed_count = 0
        
        def process_file(image_file: str):
            nonlocal processed_count
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(image_file)[0]}.txt"
            )
            
            text = self.extract_text(image_path)
            if text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    
            processed_count += 1
            if processed_count % batch_size == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Processed {processed_count}/{len(image_files)} images "
                    f"({processed_count/elapsed:.2f} images/sec)"
                )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_file, image_files)
        
        logger.info(f"Finished processing {len(image_files)} images in {time.time()-start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract text from images using OCR')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing images')
    parser.add_argument('-o', '--output', required=True, help='Output directory for text files')
    parser.add_argument('-l', '--lang', default='chi_sim', help='Language for OCR (default: chi_sim)')
    parser.add_argument('--psm', type=int, default=6, help='Page segmentation mode (default: 6)')
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocess',
                       help='Disable image preprocessing')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TextExtractor(
        lang=args.lang,
        psm=args.psm,
        preprocess=args.preprocess
    )
    
    # Process images
    extractor.process_images(
        input_dir=args.input,
        output_dir=args.output,
        max_workers=args.workers
    )

if __name__ == '__main__':
    main()
