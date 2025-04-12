from PIL import Image
import numpy as np
import brotli
import zlib
import struct
import io
import base64
import heapq

# Helper functions for decoding
def delta_decode(data):
    decoded = np.zeros_like(data, dtype=np.uint8)
    decoded[0] = data[0]
    for i in range(1, len(data)):
        decoded[i] = (decoded[i - 1] + data[i]) % 256
    return decoded

def rle_decode(encoded):
    decoded = bytearray()
    i = 0
    while i < len(encoded):
        value = encoded[i]
        count = struct.unpack(">H", encoded[i + 1:i + 3])[0]
        decoded.extend([value] * count)
        i += 3
    return bytes(decoded)

def unpack_4bit(packed_data):
    unpacked = []
    for byte in packed_data:
        first = (byte >> 4) & 0x0F
        second = byte & 0x0F
        unpacked.append(first)
        unpacked.append(second)
    return unpacked

def huffman_decode(encoded_bits, codebook):
    reverse_codebook = {v: k for k, v in codebook.items()}
    decoded = []
    current_code = ""
    for bit in encoded_bits:
        current_code += str(bit)
        if current_code in reverse_codebook:
            decoded.append(reverse_codebook[current_code])
            current_code = ""
    return bytes(decoded)

def lzw_decode(compressed):
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    decompressed = []
    w = dictionary[struct.unpack('>H', compressed[:2])[0]]
    decompressed.append(w)
    compressed = compressed[2:]
    for i in range(0, len(compressed), 2):
        k = struct.unpack('>H', compressed[i:i + 2])[0]
        if k in dictionary:
            entry = dictionary[k]
        elif k == next_code:
            entry = w + w[:1]
        else:
            raise ValueError("Invalid LZW code")
        decompressed.append(entry)
        dictionary[next_code] = w + entry[:1]
        next_code += 1
        w = entry
    return b''.join(decompressed)

def unpack_bw_1bit(packed_data, shape):
    total_pixels = shape[0] * shape[1]
    flat = []
    for byte in packed_data:
        for i in range(7, -1, -1):
            flat.append((byte >> i) & 1)
            if len(flat) == total_pixels:
                break
    return np.array(flat, dtype=np.uint8).reshape(shape)

def delta_decode_rgb(delta_array):
    delta_array = delta_array.astype(np.int16)
    decoded = np.zeros_like(delta_array, dtype=np.uint8)
    decoded[0] = delta_array[0]
    for i in range(1, len(delta_array)):
        decoded[i] = (decoded[i - 1] + delta_array[i]) % 256
    return decoded

def bitunpack_rgb(packed_data, shape):
    """
    Unpacks 5-bit-per-channel RGB data into an array of shape (height, width, 3).
    """
    height, width, _ = shape
    total_pixels = height * width
    flat = []
    buffer = 0
    bits_filled = 0

    for byte in packed_data:
        buffer = (buffer << 8) | byte
        bits_filled += 8

        while bits_filled >= 5:
            bits_filled -= 5
            val = (buffer >> bits_filled) & 0b11111  # Extract 5 bits
            flat.append(val)

            # Stop if we've unpacked enough pixels
            if len(flat) == total_pixels * 3:
                break

    # Reshape into (height, width, 3)
    return np.array(flat[:total_pixels * 3], dtype=np.uint8).reshape(height, width, 3)

# Main decoder function
def decode_lix_to_image(lix_path, output_path):
    with open(lix_path, "rb") as f:
        data = f.read()

    # Parse header
    magic = data[:4]
    if magic == b"LIXG":  # Grayscale
        width, height, method = struct.unpack(">HHB", data[4:9])
        compressed = data[9:]
        if method == 0:  # Delta only
            delta = list(compressed)
            decoded = delta_decode(delta)
        elif method == 1:  # Delta + RLE
            delta_rle = rle_decode(compressed)
            decoded = delta_decode(delta_rle)
        elif method == 2:  # Bitpack + Huffman
            raise NotImplementedError("Huffman decoding not implemented in this example.")
        elif method == 3:  # Delta + LZW
            delta_lzw = lzw_decode(compressed)
            decoded = delta_decode(delta_lzw)
        img_array = np.array(decoded, dtype=np.uint8).reshape(height, width)
        img = Image.fromarray(img_array, mode="L")
        img.save(output_path)
        print(f"✅ Decoded grayscale image saved to {output_path}")

    elif magic == b"LIXB":  # Black & White
        width, height, method = struct.unpack(">HHB", data[4:9])
        compressed = data[9:]
        if method == 1:  # Bitpack + RLE
            packed = rle_decode(compressed)
            img_array = unpack_bw_1bit(packed, (height, width))
            img = Image.fromarray(img_array * 255, mode="1")
            img.save(output_path)
            print(f"✅ Decoded B&W image saved to {output_path}")

    elif magic == b"LIXF":  # RGB
        width, height, img_type, method = struct.unpack(">HHBB", data[4:10])
        compressed = data[10:]
        if method == 1:  # Delta + Huffman
            raise NotImplementedError("Huffman decoding not implemented in this example.")
        elif method == 2:  # Bitpack + Brotli
            packed = brotli.decompress(compressed)
            rgb = bitunpack_rgb(packed, (height, width, 3))
        elif method == 3:  # Delta + Bitpack + Brotli
            packed = brotli.decompress(compressed)
            delta = bitunpack_rgb(packed, (height, width, 3))
            rgb = delta_decode_rgb(delta)
        img_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 3)
        img = Image.fromarray(img_array, mode="RGB")
        img.save(output_path)
        print(f"✅ Decoded RGB image saved to {output_path}")

    elif magic == b"LIXA":  # RGBA
        width, height, img_type, method = struct.unpack(">HHBB", data[4:10])
        rgb_compressed = data[10:]
        alpha_compressed = None
        if method != 0:
            rgb_size = len(rgb_compressed) - len(alpha_compressed)
            rgb_compressed, alpha_compressed = rgb_compressed[:rgb_size], rgb_compressed[rgb_size:]
        # Decode RGB
        if method == 1 or method == 2 or method == 3:
            packed = brotli.decompress(rgb_compressed)
            delta = bitunpack_rgb(packed, (height, width, 3))
            rgb = delta_decode_rgb(delta)
        # Decode Alpha
        if method == 0:
            alpha = np.full((height, width), 255, dtype=np.uint8)
        else:
            alpha_packed = brotli.decompress(alpha_compressed)
            alpha = delta_decode(alpha_packed)
            alpha = np.array(alpha, dtype=np.uint8).reshape(height, width)
        # Combine RGB and Alpha
        rgba = np.dstack((rgb, alpha))
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(output_path)
        print(f"✅ Decoded RGBA image saved to {output_path}")

    else:
        raise ValueError("Unsupported .lix file format!")

# Example usage
decode_lix_to_image("gray.lix", "decoded_gray.png")
decode_lix_to_image("bw.lix", "decoded_bw.png")
decode_lix_to_image("rgb.lix", "decoded_rgb.png")
decode_lix_to_image("rgba.lix", "decoded_rgba.png")