import numpy as np
from PIL import Image
import brotli
import struct
import os

def delta_encode_rgba(image_array):
    image_array = image_array.astype(np.int16)
    delta = np.zeros_like(image_array, dtype=np.uint8)
    delta[0] = image_array[0]
    delta[1:] = ((image_array[1:] - image_array[:-1]) % 256).astype(np.uint8)
    return delta

def bitpack_rgba(delta_array):
    return delta_array.astype(np.uint8).tobytes()  # No downshift!

def compress_rgba_with_hybrid(image_array):
    delta = delta_encode_rgba(image_array)
    packed = bitpack_rgba(delta)
    compressed = brotli.compress(packed)
    return compressed

def delta_encode_alpha(alpha_channel):
    alpha_channel = alpha_channel.astype(np.int16)
    delta = np.zeros_like(alpha_channel, dtype=np.uint8)
    delta[0] = alpha_channel[0]
    delta[1:] = ((alpha_channel[1:] - alpha_channel[:-1]) % 256).astype(np.uint8)
    return delta

def compress_alpha_channel(alpha_channel):
    flat_alpha = alpha_channel.flatten()
    delta_alpha = delta_encode_alpha(flat_alpha)
    compressed = brotli.compress(delta_alpha.tobytes())
    return compressed, b'\x01'

def is_opaque(alpha_channel):
    return np.all(alpha_channel == 255)

def is_strict_binary_alpha(alpha_channel):
    unique_vals = np.unique(alpha_channel)
    return np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 255])

def encode_lix_rgba(image_path):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    h, w, _ = arr.shape

    rgba_data = arr[:, :, :3].reshape(-1, 3)
    alpha_data = arr[:, :, 3]

    rgba_compressed = compress_rgba_with_hybrid(rgba_data)

    if is_opaque(alpha_data):
        alpha_compressed = b''
        alpha_method = b'\x00'
    elif is_strict_binary_alpha(alpha_data):
        alpha_flat = alpha_data.flatten()
        alpha_compressed = brotli.compress(alpha_flat.tobytes())
        alpha_method = b'\x02'
    else:
        alpha_compressed, alpha_method = compress_alpha_channel(alpha_data)

    header = (
        b'LIX' +
        b'\x04' +
        b'\x05' +
        alpha_method +
        struct.pack('>HH', w, h) +
        struct.pack('>I', len(rgba_compressed)) +
        struct.pack('>I', len(alpha_compressed))
    )

    return header + rgba_compressed + alpha_compressed


def delta_decode_rgba(packed):
    unpacked = np.frombuffer(packed, dtype=np.uint8).astype(np.int16).reshape(-1, 3)

    result = np.zeros_like(unpacked, dtype=np.int16)
    result[0] = unpacked[0]

    for i in range(1, len(unpacked)):
        result[i] = (result[i - 1] + unpacked[i]) % 256

    return result.astype(np.uint8)

def delta_decode_alpha(data):
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    result = np.zeros_like(arr, dtype=np.int16)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = (result[i - 1] + arr[i]) % 256
    return result.astype(np.uint8)

def decode_lix_rgba(lix_data):
    data = lix_data

    assert data[:3] == b'LIX', "Invalid magic bytes"
    mode = data[3]
    rgb_method = data[4]
    alpha_method = data[5]
    w, h = struct.unpack('>HH', data[6:10])
    rgb_size = struct.unpack('>I', data[10:14])[0]
    alpha_size = struct.unpack('>I', data[14:18])[0]

    rgb_compressed = data[18:18 + rgb_size]
    alpha_compressed = data[18 + rgb_size:18 + rgb_size + alpha_size]

    rgb_packed = brotli.decompress(rgb_compressed)
    rgb_array = delta_decode_rgba(rgb_packed).reshape(h, w, 3)

    if alpha_method == 0:
        alpha_channel = np.full((h * w,), 255, dtype=np.uint8)
    elif alpha_method == 1:
        decoded = brotli.decompress(alpha_compressed)
        alpha_flat = delta_decode_alpha(decoded)
        alpha_channel = alpha_flat
    elif alpha_method == 2:
        alpha_flat = brotli.decompress(alpha_compressed)
        alpha_channel = np.frombuffer(alpha_flat, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown alpha method: {alpha_method}")

    alpha_channel = alpha_channel.reshape(h, w)

    rgba = np.dstack([rgb_array, alpha_channel])
    return Image.fromarray(rgba, mode="RGBA")


def save_decoded_image(image_path, output_path, save_intermediate=False):
    """
    Encodes, decodes, and saves an RGBA image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the decoded image.
        save_intermediate (bool): If True, saves the .lix file.
    """
    try:
        # Encode the image
        lix_data = encode_lix_rgba(image_path)

        # Save intermediate .lix file if requested
        lix_file_path = os.path.splitext(output_path)[0] + ".lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as {lix_file_path}, size: {len(lix_data)} bytes")
        else:
            print("✅ .lix file not saved (save_intermediate=False)")

        # Decode the image
        decoded_img = decode_lix_rgba(lix_data)

        # Save the decoded image
        decoded_img.save(output_path)
        print(f"✅ RGBA image decoded and saved as {output_path}")

        # Compression Ratio Calculation
        original_size = os.path.getsize(image_path)
        compressed_size = len(lix_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        print(f"📊 Compression Ratio: {compression_ratio:.2f} (Original Size: {original_size} bytes, Compressed Size: {compressed_size} bytes)")


    except Exception as e:
        print(f"An error occurred: {e}")
