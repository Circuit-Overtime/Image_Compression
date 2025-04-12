# 📦 .lix Image Compression System

A custom image compression pipeline designed for efficient storage and transmission using a novel `.lix` format. The system supports multiple image types — **Strict Black & White**, **Grayscale**, **RGB**, and **RGBA** — and applies smart compression strategies based on image variance and content.

---

## 🚀 Features

- **Supported Image Types**:
    - 🖤 **Strict Black & White (1-bit)**
    - 🌫️ **Grayscale (8-bit)**
    - 🌈 **RGB**
    - 🎨 **RGBA (with transparency)**
- **Compression Techniques**:
    - Delta Encoding
    - Run-Length Encoding (RLE)
    - Huffman Coding
    - LZW Compression
    - Brotli Compression (optional)
- **Key Capabilities**:
    - Automatic classification of input images
    - Smart compression strategy selection based on image content
    - Export of `.lix` files and reconstructed images
    - Benchmarking of compression ratio and speed
    - Support for both encoding and decoding
    - CLI compatibility and folder-wide benchmarking

---

## 📁 Project Structure

```
├── master_driver.py          # Main pipeline entry point
├── benchmark.py              # Benchmark script for folders
├── test_images/              # Input images organized in subfolders
├── enc_op/                   # Output directory for .lix and decompressed files
├── master_encoder_bw.py      # Strict Black & White encoder/decoder
├── master_encoder_gray.py    # Grayscale encoder/decoder
├── master_encoder_rgb.py     # RGB encoder/decoder
├── master_encoder_rgba.py    # RGBA encoder/decoder
├── utils/                    # (Optional) Shared utilities
├── README.md                 # This file
```


---

## ⚙️ How It Works

1. **Input Image** ➜
2. **Classified** as one of: BW / Grayscale / RGB / RGBA ➜
3. **Encoded** using optimized compression pipelines ➜
4. **.lix File** is created ➜
5. **Decoded Output** is reconstructed and saved ➜
6. **Benchmarking** evaluates compression efficiency

---

## 🧪 Running the Project

### 🔹 1. Encode and Decode a Single Image

Run the master driver (example):

```bash
python master_driver.py --input ./test_images/sample.png --output ./enc_op/
```
> This will generate a .lix file and a decompressed version for comparison.

### 🔹 2. Benchmark a Folder of Images

```python
python benchmark.py
```

> All supported images in `./test_images/` will be processed. Results are saved in `benchmark_results.csv`.

### 🔹 3. Classify an Image Type (Internally Used)

```python
from PIL import Image
from classify import classify_image_type

image_type = classify_image_type("path/to/image.png")
print(image_type)  # Outputs: BW, Grayscale, RGB, or RGBA
```
----

# 📊 Benchmark Output

`benchmark_results.csv` contains:

| Filename | Category | Type | Original\_KB | Decoded\_KB | Time(s) |
| --- | --- | --- | --- | --- | --- |
| grayscale\_1.jpg | Grayscale | G | 12.5 | 84.1 | 0.034 |

----

# 📌 Compression Strategies Used

**Each encoder chooses the best available strategy based on content**:

- **BW**: Bit-packing, RLE, Huffman
- **Grayscale**: Delta + Huffman / LZW
- **RGB: Delta + RLE + Huffman**, Channel-wise split
- **RGBA**: RGB compression + Alpha channel RLE

The best compression method is automatically chosen based on statistical analysis (variance, entropy).

# ✅ Requirements

- **Python 3.7+**
- **Pillow (pip install pillow)**
- **NumPy (pip install numpy)**

> ## **Optional: Brotli support requires pip install brotli**


# 🧠 Notes

- **.lix is a custom intermediate binary format, not a standard image format.**
- **This system is designed for experimentation and research.**
- **Not compatible with web browsers or image viewers directly.**
- **Focus is on compression ratio and speed over compatibility.**

# ✨ Future Work

- **Add GUI for drag-and-drop compression**

- **Add support for animated images (GIFs)**

- **Extend to video frame compression**

- **Explore neural network–assisted lossless compression**

# 👨‍💻 Author

- ## Developed by [Ayushman Bhattacharya], 2025
- ## Inspired by the need for efficient, flexible, and intelligent image compression systems.

# 📄 License
**MIT License – use freely, credit appreciated!**





