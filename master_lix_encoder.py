from PIL import Image
import numpy as np
import brotli
import zlib
import struct
import io
import base64
import os
import heapq
from collections import Counter



COMPRESSION_METHODS = {
    "grayscale": {
        "delta": 0x01,
        "rle": 0x02,
        "bitpack": 0x03,
        "huffman": 0x04,
        "brotli": 0x05,
        "lzw": 0x06,
    },
    "bw": {
        "bitpack": 0x11,
        "huffman": 0x12,
    },
    "rgb": {
        "rle": 0x21,
        "huffman": 0x22,
        "brotli": 0x23,
        "lzw": 0x24,
    },
    "rgba": {
        "delta": 0x31,
        "rle": 0x32,
        "huffman": 0x33,
        "brotli": 0x34,
        "lzw": 0x35,
    }
}


def delta_encode(data):
    delta = [data[0]]
    for i in range(1, len(data)):
        delta.append(np.uint8(data[i] - data[i - 1]))
    return delta

def rle_encode(data):
    encoded = bytearray()
    i = 0
    while i < len(data):
        count = 1
        while i + count < len(data) and data[i] == data[i + count] and count < 65535:
            count += 1
        encoded.append(data[i])                     
        encoded.extend(struct.pack(">H", count))    
        i += count
    return encoded

def bitpack_4bit(data):
    packed = []
    for i in range(0, len(data), 2):
        first = data[i] & 0x0F
        second = data[i+1] & 0x0F if i+1 < len(data) else 0
        packed.append((first << 4) | second)
    return packed

def huffman_encode(data):
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

def lzw_encode(data):
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

### 🔧 Grayscale Encoder Pipeline
def encode_grayscale_lix(image_path):
    img = Image.open(image_path).convert("L") # Removed resize()
    data = np.array(img).flatten()
    width, height = img.size

    methods = {}
    
    # Method 0: Delta only
    delta = delta_encode(data)
    methods[0] = bytes(delta)

    # Method 1: Delta + RLE
    delta_rle = rle_encode(delta)
    methods[1] = bytes(delta_rle)

    # Method 2: Bit-pack + Huffman (only if range fits 0-15)
    if np.all(data < 16):
        bitpacked = bitpack_4bit(data)
        huff = huffman_encode(bitpacked)
        methods[2] = huff

    # Method 3: Delta + LZW
    delta_lzw = lzw_encode(delta)
    methods[3] = delta_lzw

    # Choose smallest
    best_method = min(methods, key=lambda k: len(methods[k]))
    compressed = methods[best_method]

    # Construct .lix binary
    header = b'LIXG'
    meta = struct.pack('>HHB', width, height, best_method)
    return header + meta + compressed

def pack_bw_1bit(image_array):
    flat = image_array.flatten()
    packed = []

    for i in range(0, len(flat), 8):
        byte = 0
        for j in range(8):
            if i + j < len(flat):
                byte |= (flat[i + j] & 1) << (7 - j)
        packed.append(byte)

    return bytes(packed)

def unpack_bw_1bit(packed_data, shape):
    total_pixels = shape[0] * shape[1]
    flat = []

    for byte in packed_data:
        for i in range(7, -1, -1):
            flat.append((byte >> i) & 1)
            if len(flat) == total_pixels:
                break

    return np.array(flat, dtype=np.uint8).reshape(shape)


def rle_decode(encoded):
    decoded = []
    i = 0
    while i < len(encoded):
        value = encoded[i]
        count = struct.unpack(">H", encoded[i + 1:i + 3])[0]
        decoded.extend([value] * count)
        i += 3
    return bytes(decoded)


def calculate_image_variance(img_arr):
    return np.var(img_arr)

def delta_encode_rgb(img_array):
    delta = np.zeros_like(img_array)
    delta[0] = img_array[0]
    delta[1:] = (img_array[1:].astype(np.int16) - img_array[:-1].astype(np.int16)) % 256
    return delta.astype(np.uint8)


### --- Huffman Compression --- ###
class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [Node(char=k, freq=v) for k, v in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(freq=n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)
    return heap[0] if heap else None

def build_huffman_dict(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node:
        if node.char is not None:
            codebook[node.char] = prefix
        build_huffman_dict(node.left, prefix + "0", codebook)
        build_huffman_dict(node.right, prefix + "1", codebook)
    return codebook

def compress_huffman(data):
    tree = build_huffman_tree(data)
    codebook = build_huffman_dict(tree)
    encoded_bits = ''.join(codebook[byte] for byte in data)
    padding = 8 - len(encoded_bits) % 8
    encoded_bits += '0' * padding
    b = bytearray()
    for i in range(0, len(encoded_bits), 8):
        b.append(int(encoded_bits[i:i+8], 2))
    return bytes(b)

### --- Compression Strategies --- ###

def compress_delta_huffman(img_arr):
    delta = delta_encode_rgb(img_arr)
    flat = delta.flatten().tobytes()
    return compress_huffman(flat)

def compress_bitpack_brotli(img_arr):
    packed = bitpack_rgb(img_arr)
    return brotli.compress(packed)

def compress_delta_bitpack_brotli(img_arr):
    delta = delta_encode_rgb(img_arr)
    packed = bitpack_rgb(delta)
    return brotli.compress(packed)

### --- Final Smart Encoder --- ###

def encode_lix_rgb(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img).reshape(-1, 3)
    height, width = img.height, img.width

    variance = calculate_image_variance(img_arr)
    
    # Compression selector based on image complexity
    if variance < 300:
        method = 1  # Delta + Huffman
        compressed = compress_delta_huffman(img_arr)
    elif variance < 800:
        method = 2  # Bitpack + Brotli
        compressed = compress_bitpack_brotli(img_arr)
    else:
        method = 3  # Delta + Bitpack + Brotli
        compressed = compress_delta_bitpack_brotli(img_arr)

    # Header Format: [magic (4 bytes)] [width (2)] [height (2)] [type (1=RGB)] [method (1)]
    header = b'LIXF' + struct.pack(">HHBB", width, height, 0x03, method)
    with open(output_path, "wb") as f:
        f.write(header + compressed)
        print(f"✅ Saved .lix RGB: {output_path}, method: {method}, size: {len(compressed)} bytes")


def delta_encode_rgb(image_array):
    image_array = image_array.astype(np.int16)  # Upgrade to avoid overflow
    delta = np.zeros_like(image_array, dtype=np.uint8)
    delta[0] = image_array[0]
    delta[1:] = ((image_array[1:] - image_array[:-1]) % 256).astype(np.uint8)
    return delta


def bitpack_rgb(delta_array):
    import numpy as np

    flat = delta_array.reshape(-1, 3).astype(np.int32)

    packed = bytearray()
    buffer = 0  # native Python int
    bits_filled = 0

    for pixel in flat:
        for channel in pixel:
            val = int(channel)  # absolute conversion to Python int
            val = (val + 256) % 256  # wrap-around for deltas
            bits = val & 0b11111     # restrict to 5 bits

            buffer = (int(buffer) << 5) | int(bits)
            bits_filled = int(bits_filled) + 5

            while bits_filled >= 8:
                bits_filled = int(bits_filled) - 8
                byte = (int(buffer) >> bits_filled) & 0xFF
                packed.append(int(byte))  # hard enforce int

    if bits_filled > 0:
        print("buffer type:", type(buffer), "bits_filled type:", type(bits_filled))
        byte = (int(buffer) << (8 - bits_filled)) & 0xFF
        packed.append(int(byte))

    return bytes(packed)




def compress_rgb_with_hybrid(image_array):
    delta = delta_encode_rgb(image_array)
    packed = bitpack_rgb(delta)
    compressed = brotli.compress(packed)
    return compressed

def delta_encode_alpha(alpha_channel):
    alpha_channel = alpha_channel.astype(np.int16)
    delta = np.zeros_like(alpha_channel, dtype=np.uint8)
    delta[0] = alpha_channel[0]
    delta[1:] = ((alpha_channel[1:] - alpha_channel[1:]) % 256).astype(np.uint8)
    return delta


def compress_alpha_channel(alpha_channel):
    flat_alpha = alpha_channel.flatten()
    delta_alpha = delta_encode_alpha(flat_alpha)
    compressed = brotli.compress(delta_alpha.tobytes())
    return compressed, 1

def is_opaque(alpha_channel):
    return np.all(alpha_channel == 255)

def is_strict_binary_alpha(alpha_channel):
    unique_vals = np.unique(alpha_channel)
    return np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 255])


def save_lix_image(image_path, output_path):
    img = Image.open(image_path)
    mode = img.mode
    width, height = img.size

    if mode == "L":
        # Grayscale
        header = b"LIXG"
        data = np.array(img).flatten()
        delta = delta_encode(data)
        delta_rle = rle_encode(delta)
        lzw = lzw_encode(delta)
        methods = {
            0: bytes(delta),
            1: bytes(delta_rle),
            3: lzw,
        }
        if np.all(data < 16):
            methods[2] = huffman_encode(bitpack_4bit(data))
        best_method = min(methods, key=lambda k: len(methods[k]))
        compressed = methods[best_method]
        meta = struct.pack('>HHB', width, height, best_method)
        with open(output_path, "wb") as f:
            f.write(header + meta + compressed)
        print(f"✅ Saved .lix Grayscale: {output_path}, size: {len(compressed)} bytes")

    elif mode == "1":
        # Strict Black & White
        img = img.convert("1").resize((width, height))
        arr = np.array(img).astype(np.uint8)
        packed = pack_bw_1bit(arr)
        rle = rle_encode(packed)
        header = b"LIXB"
        meta = struct.pack(">HHB", width, height, 1)
        with open(output_path, "wb") as f:
            f.write(header + meta + rle)
        print(f"✅ Saved .lix B&W: {output_path}, size: {len(rle)} bytes")

    elif mode == "RGB":
        # RGB Smart Compression
        img = img.convert("RGB")
        arr = np.array(img).reshape(-1, 3)
        variance = np.var(arr)

        if variance < 300:
            method = 1  # Delta + Huffman
            compressed = compress_delta_huffman(arr)
        elif variance < 800:
            method = 2  # Bitpack + Brotli
            compressed = compress_bitpack_brotli(arr)
        else:
            method = 3  # Delta + Bitpack + Brotli
            compressed = compress_delta_bitpack_brotli(arr)

        header = b"LIXF" + struct.pack(">HHBB", width, height, 0x03, method)
        with open(output_path, "wb") as f:
            f.write(header + compressed)
        print(f"✅ Saved .lix RGB: {output_path}, method: {method}, size: {len(compressed)} bytes")

    elif mode == "RGBA":
        # RGBA Split Compression
        img = img.convert("RGBA")
        arr = np.array(img)
        rgb = arr[:, :, :3].reshape(-1, 3)
        alpha = arr[:, :, 3]

        rgb_compressed = compress_rgb_with_hybrid(rgb)

        if is_opaque(alpha):
            alpha_compressed = b''  # No alpha needed
            method = 0
        else:
            alpha_compressed, method = compress_alpha_channel(alpha)

        header = b"LIXA" + struct.pack(">HHBB", width, height, 0x04, method)
        with open(output_path, "wb") as f:
            f.write(header + rgb_compressed + alpha_compressed)
        print(f"✅ Saved .lix RGBA: {output_path}, method: {method}, total size: {len(rgb_compressed) + len(alpha_compressed)} bytes")


    else:
        raise ValueError("Unsupported image mode for .lix format!")

def lix_to_base64(lix_data):
    return base64.b64encode(lix_data).decode('utf-8')

def base64_to_lix(b64_string, output_path=None):
    decoded = base64.b64decode(b64_string)
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(decoded)
        print(f"✅ Decoded .lix file saved to {output_path}")
    return decoded


def encode_image_to_base64(image_path):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".lix") as tmp:
        temp_path = tmp.name
    save_lix_image(image_path, temp_path)
    with open(temp_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    os.remove(temp_path)
    return encoded


save_lix_image("exp2.jpg", "gray.lix")
save_lix_image("exp3.png", "bw.lix")
save_lix_image("exp4.jpg", "rgb.lix")
save_lix_image("exp_rgba.png", "rgba.lix")