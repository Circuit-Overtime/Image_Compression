import os
import time
from PIL import Image
from master_encoder_gray import save_decoded_image as save_decoded_image_gray
from master_encoder_rgb import save_decoded_image as save_decoded_image_rgb
from master_encoder_sbw import save_decoded_image as save_decoded_image_bw
from master_encoder_rgba import save_decoded_image as save_decoded_image_rgba
import numpy as np

def classify_image_type(image_path):
    img = Image.open(image_path)
    mode = img.mode
    if mode == '1':
        return 'BW'
    elif mode == 'L':
        arr = np.array(img)
        unique_vals = np.unique(arr)
        if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
            return 'BW'
        return 'Grayscale'
    elif mode == 'RGBA':
        return 'RGBA'
    elif mode == 'RGB':
        arr = np.array(img)
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            unique_vals = np.unique(arr[:, :, 0])
            if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
                return 'BW'
            return 'Grayscale'
        return 'RGB'
    else:
        return 'Unknown'

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024.0

def benchmark_folder(image_folder, output_csv="benchmark_results.csv"):
    results = []
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        input_path = os.path.join(image_folder, filename)
        image_type = classify_image_type(input_path)
        name, _ = os.path.splitext(filename)
        lix_path = f"enc_op/{name}.lix"
        out_path = f"enc_op/{name}_decompressed.png" if image_type == "RGBA" else f"enc_op/{name}_decompressed.jpg"

        start_enc = time.time()
        if image_type == 'BW':
            save_decoded_image_bw(input_path, out_path, save_intermediate=True)
        elif image_type == 'Grayscale':
            save_decoded_image_gray(input_path, out_path, save_intermediate=True)
        elif image_type == 'RGB':
            save_decoded_image_rgb(input_path, out_path, save_intermediate=True)
        elif image_type == 'RGBA':
            save_decoded_image_rgba(input_path, out_path, save_intermediate=True)
        else:
            print(f"❌ Skipping unsupported image: {filename}")
            continue
        end_enc = time.time()

        original_size = get_file_size_kb(input_path)
        lix_size = get_file_size_kb(lix_path)
        output_size = get_file_size_kb(out_path)
        compression_time = round(end_enc - start_enc, 3)

        results.append([filename, image_type, original_size, lix_size, output_size, compression_time])

    # Write CSV
    with open(output_csv, "w") as f:
        f.write("Filename,Type,Original_KB,.lix_KB,Decompressed_KB,Time_sec\n")
        for r in results:
            f.write(",".join(map(str, r)) + "\n")

    print(f"✅ Benchmark complete. Results saved to {output_csv}")
