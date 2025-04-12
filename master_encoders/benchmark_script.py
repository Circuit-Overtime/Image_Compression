import os
import time
from PIL import Image
from master_encoder_gray import save_decoded_image as save_decoded_image_gray
from master_encoder_rgb import save_decoded_image as save_decoded_image_rgb
from master_encoder_sbw import save_decoded_image as save_decoded_image_bw
from master_encoder_rgba import save_decoded_image as save_decoded_image_rgba
import numpy as np

def classify_image_type(image_path):
    try:
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
    except Exception as e:
        print(f"❌ Error classifying image {image_path}: {e}")
        return 'Unknown'

def get_file_size_kb(path):
    try:
        return os.path.getsize(path) / 1024.0
    except FileNotFoundError:
        return 0.0  # Or some other sentinel value

def benchmark_folder(image_folder, output_csv="benchmark_results.csv"):
    results = []
    os.makedirs("enc_op/benchmark_test", exist_ok=True)  # Create the output directory if it doesn't exist

    # Create subfolders in enc_op corresponding to the test_images subfolders
    for root, dirs, files in os.walk(image_folder):
        for dirname in dirs:
            os.makedirs(os.path.join("enc_op/benchmark_test", dirname), exist_ok=True)

    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            input_path = os.path.join(root, filename)

            image_type = classify_image_type(input_path)
            name, _ = os.path.splitext(filename)
            relative_path = os.path.relpath(root, image_folder)
            out_path = os.path.join("enc_op/benchmark_test", relative_path, f"{name}_decompressed.png" if image_type == "RGBA" else f"{name}_decompressed.jpg")
            print(input_path)

            start_enc = time.time()
            output_size = 0.0  # Initialize before the try block

            try:
                if image_type == 'BW':
                    save_decoded_image_bw(input_path, out_path, save_intermediate=False)
                elif image_type == 'Grayscale':
                    save_decoded_image_gray(input_path, out_path, save_intermediate=False)
                elif image_type == 'RGB':
                    save_decoded_image_rgb(input_path, out_path, save_intermediate=False)
                elif image_type == 'RGBA':
                    save_decoded_image_rgba(input_path, out_path, save_intermediate=False)
                else:
                    print(f"❌ Skipping unsupported image: {filename}")
                    continue

                # Wait for the file to be created (up to a limit) and THEN get the size
                start_wait = time.time()
                wait_time = 0
                while not os.path.exists(out_path) and wait_time < 5:  # Wait for up to 5 seconds
                    time.sleep(0.1) # Sleep for 100ms
                    wait_time = time.time() - start_wait

                if os.path.exists(out_path):
                    output_size = get_file_size_kb(out_path)
                else:
                    print(f"❌ Decompressed image not found after waiting: {out_path}")


            except Exception as e:
                print(f"❌ Error encoding/decoding {filename}: {e}")


            end_enc = time.time()
            original_size = get_file_size_kb(input_path)
            compression_time = round(end_enc - start_enc, 3)
            category = os.path.basename(root)

            results.append([filename, category, image_type, original_size, output_size, compression_time])  
            print(f"✅ Processed {filename}: Original: {original_size:.2f} KB, Decompressed: {output_size:.2f} KB, Time: {compression_time:.3f} sec")


    with open(output_csv, "w") as f:
        f.write("Filename,Category,Type,Original_KB,Decompressed_KB,Time_sec\n") # Include Category
        for r in results:
            f.write(",".join(map(str, r)) + "\n")

    print(f"✅ Benchmark complete. Results saved to {output_csv}")


benchmark_folder("./test_images", output_csv="benchmark_results.csv")