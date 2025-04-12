import numpy as np
from PIL import Image
import base64
import struct
import heapq
import io
import os

### Compression utilities
def delta_encode_gray(data):
    delta = [data[0]]
    for i in range(1, len(data)):
        delta.append(np.uint8(data[i] - data[i - 1]))
    return delta

def rle_encode_gray(data):
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1] and count < 255:
            count += 1
        else:
            encoded.extend([data[i-1], count])
            count = 1
    encoded.extend([data[-1], count])
    return encoded

def bitpack_4bit_gray(data):
    packed = []
    for i in range(0, len(data), 2):
        first = data[i] & 0x0F
        second = data[i+1] & 0x0F if i+1 < len(data) else 0
        packed.append((first << 4) | second)
    return packed

def huffman_encode_gray(data):
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    heap = [[weight, [sym, ""]] for sym, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huff_map = {sym: code for sym, code in heap[0][1:]}
    encoded_bits = ''.join([huff_map[b] for b in data])
    padded = encoded_bits + '0' * ((8 - len(encoded_bits) % 8) % 8)
    return bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))

def lzw_encode_gray(data):
    dictionary = {bytes([i]): i for i in range(256)}
    w = b""
    codes = []
    for byte in data:
        wc = w + bytes([byte])
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = len(dictionary)
            w = bytes([byte])
    if w:
        codes.append(dictionary[w])
    encoded_bytes = b''.join(struct.pack('>H', code) for code in codes)
    return encoded_bytes

### Decompression utilities
def delta_decode_grey(data):
    decoded = [data[0]]
    for i in range(1, len(data)):
        decoded.append((decoded[-1] + data[i]) % 256)
    return decoded

def rle_decode_grey(data):
    decoded = []
    for i in range(0, len(data), 2):
        value = data[i]
        count = data[i + 1]
        decoded.extend([value] * count)
    return decoded

def bitunpack_4bit_grey(data):
    unpacked = []
    for byte in data:
        unpacked.append((byte >> 4) & 0x0F)
        unpacked.append(byte & 0x0F)
    return unpacked

def huffman_decode_grey(encoded_data, huff_map):
    reverse_map = {v: k for k, v in huff_map.items()}
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_data)
    decoded = []
    buffer = ""
    for bit in bit_string:
        buffer += bit
        if buffer in reverse_map:
            decoded.append(reverse_map[buffer])
            buffer = ""
    return decoded

def lzw_decode_grey(encoded_data):
    codes = [struct.unpack('>H', encoded_data[i:i + 2])[0] for i in range(0, len(encoded_data), 2)]
    dictionary = {i: bytes([i]) for i in range(256)}
    w = bytes([codes[0]])
    decoded = [w]
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == len(dictionary):
            entry = w + w[:1]
        else:
            raise ValueError("Invalid LZW code")
        decoded.append(entry)
        dictionary[len(dictionary)] = w + entry[:1]
        w = entry
    return b''.join(decoded)


### 🔧 Grayscale Encoder Pipeline
def encode_grayscale_lix_gray(img):  # Changed: takes image object as input
    data = np.array(img).flatten()
    width, height = img.size

    methods = {}
    
    # Method 0: Delta only
    delta = delta_encode_gray(data)
    methods[0] = bytes(delta)

    # Method 1: Delta + RLE
    delta_rle = rle_encode_gray(delta)
    methods[1] = bytes(delta_rle)

    # Method 2: Bit-pack + Huffman (only if range fits 0-15)
    if np.all(data < 16):
        bitpacked = bitpack_4bit_gray(data)
        huff = huffman_encode_gray(bitpacked)
        methods[2] = huff

    # Method 3: Delta + LZW
    delta_lzw = lzw_encode_gray(delta)
    methods[3] = delta_lzw

    # Choose smallest
    best_method = min(methods, key=lambda k: len(methods[k]))
    compressed = methods[best_method]

    # Construct .lix binary
    header = b'LIXG'
    meta = struct.pack('>HHB', width, height, best_method)
    return header + meta + compressed


### Grayscale Decoder Pipeline
def decode_grayscale_lix_grey(lix_data): # Changed: takes lix data (bytes) as input
    # Parse header and metadata
    header = lix_data[:4]
    if header != b'LIXG':
        raise ValueError("Invalid .lix file format")
    
    width, height, method = struct.unpack('>HHB', lix_data[4:9])
    compressed_data = lix_data[9:]

    # Decompress based on method
    if method == 0:  # Delta only
        decompressed = delta_decode_grey(compressed_data)
    elif method == 1:  # Delta + RLE
        delta_rle = rle_decode_grey(compressed_data)
        decompressed = delta_decode_grey(delta_rle)
    elif method == 2:  # Bit-pack + Huffman
        # Note: Huffman decoding requires the Huffman map, which is not stored in the .lix file.
        # This implementation assumes the map is predefined or stored elsewhere.
        raise NotImplementedError("Huffman decoding requires the Huffman map")
    elif method == 3:  # Delta + LZW
        delta_lzw = lzw_decode_grey(compressed_data)
        decompressed = delta_decode_grey(delta_lzw)
    else:
        raise ValueError("Unknown compression method")

    # Reshape and return as image
    decompressed = np.array(decompressed, dtype=np.uint8).reshape((height, width))
    return Image.fromarray(decompressed, mode="L")


### Master function to encode, decode and save image
def save_decoded_image(image_path, output_path, save_intermediate=False):
    """
    Encodes, decodes, and saves an image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the decoded image.
        save_intermediate (bool): If True, saves the .lix file.
    """
    try:
        # Load the image
        img = Image.open(image_path).convert("L").resize((128, 128))

        # Encode the image
        lix_data = encode_grayscale_lix_gray(img)

        # Save intermediate .lix file if requested
        lix_file_path = os.path.splitext(output_path)[0] + ".lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as {lix_file_path}, size: {len(lix_data)} bytes")
            # Compare the original and decoded images
            original_size = os.path.getsize(image_path)
            decoded_size = os.path.getsize(output_path)
            compression_ratio = original_size / len(lix_data)

            print(f"✅ Original image size: {original_size} bytes")
            print(f"✅ Decoded image size: {decoded_size} bytes")
            print(f"✅ Compression ratio: {compression_ratio:.2f}")
        else:
            print("✅ .lix file not saved (save_intermediate=False)")

        # Decode the image
        decoded_img = decode_grayscale_lix_grey(lix_data)

        # Save the decoded image
        decoded_img.save(output_path)
        print(f"✅ Decoded image saved as {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

  