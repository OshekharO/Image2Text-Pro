# Image2Text-Pro

Advanced OCR tool for extracting text from images with preprocessing and parallel processing.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-orange)

## Features ✨

- 📷 Supports multiple image formats (JPG, PNG, TIFF, BMP)
- 🔍 Advanced image preprocessing for better OCR accuracy
- ⚡ Parallel processing for fast batch operations
- 🌍 Multi-language support (Chinese by default)
- 📊 Progress tracking and performance metrics
- 🛠️ Configurable preprocessing and OCR parameters

## Installation 🛠️

1. Install Tesseract OCR:
   ```bash
   # On Ubuntu/Debian
   sudo apt install tesseract-ocr
   sudo apt install libtesseract-dev

   # On macOS
   brew install tesseract
   ```

2. Python Dependencies:
   ```python
    pip install -r requirements.txt
   ```

## Usage 🚀

1. Basic Command:
   
   `
   python program.py -i input_images -o output_texts
   `

3. Advanced Usage:
   
   `
   python program.py \
  -i ./photos \
  -o ./extracted_texts \
  --lang eng+chi_sim \
  --psm 11 \
  --workers 8 
   `
   
## Contributing 🤝
   We welcome contributions! Please:
   1. Fork the repository
   2. Create a feature branch (git checkout -b feature/your-feature)
   3. Commit your changes (git commit -m 'Add some feature')
   4. Push to the branch (git push origin feature/your-feature)
   5. Open a Pull Request

<div align="center"> <p>Made with ❤️ and Python</p> <sub>OCR accuracy may vary depending on image quality and language complexity</sub> </div>
