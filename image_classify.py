from PIL import Image
import numpy as np
import os

def classify_image_type(image_path):
    img = Image.open(image_path)
    mode = img.mode

    if mode == '1':
        return 'BW'  # Strict black and white
    elif mode == 'L':
        return 'Grayscale'
    elif mode == 'RGB':
        return 'RGB'
    elif mode == 'RGBA':
        return 'RGBA'
    else:
        # Heuristic for strict BW: all pixel values are either 0 or 255
        arr = np.array(img.convert('L'))
        unique_vals = np.unique(arr)
        if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
            return 'BW'
        return 'Unknown'


# def process_pipeline(image_path):
    img_type = classify_image_type(image_path)
    print(f"🔍 Detected Image Type: {img_type}")

    base = os.path.splitext(image_path)[0]
    lix_path = base + ".lix"
    out_path = base + "_decompressed.jpg" if img_type == 'RGB' else base + "_decompressed.png"

    if img_type == 'RGB':
        from lix_rgb import encode_lix_rgb, decode_lix_rgb
        encode_lix_rgb(image_path, lix_path)
        decode_lix_rgb(lix_path, out_path)

    elif img_type == 'Grayscale':
        from lix_gray import encode_lix_grayscale, decode_lix_grayscale
        encode_lix_grayscale(image_path, lix_path)
        decode_lix_grayscale(lix_path, out_path)

    elif img_type == 'RGBA':
        from lix_rgba import encode_lix_rgba, decode_lix_rgba
        encode_lix_rgba(image_path, lix_path)
        decode_lix_rgba(lix_path, out_path)

    elif img_type == 'BW':
        from lix_bw import encode_lix_bw, decode_lix_bw
        encode_lix_bw(image_path, lix_path)
        decode_lix_bw(lix_path, out_path)

    else:
        raise ValueError("Unsupported or unknown image format")

    print(f"✅ Done! Decompressed image saved at: {out_path}")


# Example usage:
type = classify_image_type("exp_bw.jpg")
print(type)
