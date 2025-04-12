from PIL import Image
import numpy as np
import base64, zlib, brotli
import zstandard as zstd
import lzma, io
from collections import defaultdict, Counter
import heapq

# === Basic Utilities ===
def flatten_channels(image):
    R, G, B = image[:,:,0].flatten(), image[:,:,1].flatten(), image[:,:,2].flatten()
    return R, G, B

def delta_encode(data):
    return [data[0]] + [np.uint8(data[i] - data[i-1]) for i in range(1, len(data))]

def delta_decode(data):
    out = [data[0]]
    for i in range(1, len(data)):
        out.append(np.uint8((out[-1] + data[i]) % 256))
    return out

def rle_encode(data):
    encoded = []
    prev, count = data[0], 1
    for val in data[1:]:
        if val == prev and count < 255:
            count += 1
        else:
            encoded.extend([count, prev])
            prev, count = val, 1
    encoded.extend([count, prev])
    return bytes(encoded)

def bitpack_4bit_channels(channels):
    packed = []
    for ch in channels:
        data = (ch // 16).astype(np.uint8)  # Reduce to 4 bits
        for i in range(0, len(data), 2):
            first = data[i]
            second = data[i+1] if i+1 < len(data) else 0
            packed.append((first << 4) | second)
    return bytes(packed)

# === Huffman ===
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encode(data):
    freq = Counter(data)
    heap = [HuffmanNode(byte, freq[byte]) for byte in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        merged = HuffmanNode(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)

    codes = {}
    def build_codes(node, code=''):
        if node.char is not None:
            codes[node.char] = code
            return
        build_codes(node.left, code + '0')
        build_codes(node.right, code + '1')
    build_codes(heap[0])

    encoded_bits = ''.join(codes[b] for b in data)
    return encoded_bits

# === LZW ===
def lzw_compress(data):
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = b""
    result = []
    for byte in data:
        wc = w + bytes([byte])
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([byte])
    if w:
        result.append(dictionary[w])
    return result

# === LZ77 (basic) ===
def lz77_compress(data, window_size=128):
    i, result = 0, []
    while i < len(data):
        match_len, match_distance = 0, 0
        for j in range(max(0, i - window_size), i):
            l = 0
            while (i + l < len(data)) and (data[j + l] == data[i + l]):
                l += 1
            if l > match_len:
                match_len, match_distance = l, i - j
        if match_len > 2:
            result.append((1, match_distance, match_len))
            i += match_len
        else:
            result.append((0, data[i]))
            i += 1
    return result

# === LZSS ===
def lzss_compress(data, window_size=128):
    i, result = 0, []
    while i < len(data):
        match_len, match_distance = 0, 0
        for j in range(max(0, i - window_size), i):
            l = 0
            while i + l < len(data) and data[j + l] == data[i + l]:
                l += 1
            if l > match_len:
                match_len, match_distance = l, i - j
        if match_len >= 3:
            result.append((1, match_distance, match_len))
            i += match_len
        else:
            result.append((0, data[i]))
            i += 1
    return result

# === Run Compression Benchmark ===
def benchmark_image(image_path):
    image = Image.open(image_path).resize((128, 128)).convert("RGB")
    pixels = np.array(image)
    R, G, B = flatten_channels(pixels)
    all_channels = np.concatenate([R, G, B])

    results = []

    # Original
    original_b64 = base64.b64encode(all_channels).decode()
    results.append(("Original (RGB base64)", len(original_b64)))

    # Delta per channel
    delta_all = np.concatenate([
        delta_encode(R),
        delta_encode(G),
        delta_encode(B)
    ])
    results.append(("Delta (RGB)", len(delta_all)))

    # Delta + RLE
    rle_delta_all = rle_encode(delta_all)
    results.append(("Delta + RLE (RGB)", len(rle_delta_all)))

    # 4-bit packing
    packed = bitpack_4bit_channels([R, G, B])
    results.append(("BitPack (4-bit RGB)", len(packed)))

    # BitPack + Huffman
    huff_bits = huffman_encode(packed)
    huff_bytes = (len(huff_bits) + 7) // 8
    results.append(("BitPack + Huffman", huff_bytes))

    # LZW (on grayscale equivalent)
    gray = np.dot(pixels[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8).flatten()
    lzw = lzw_compress(gray.tobytes())
    results.append(("LZW (Grayscale)", len(lzw)))

    # Delta + LZW
    delta_gray = delta_encode(gray)
    lzw_delta = lzw_compress(bytes(delta_gray))
    results.append(("Delta + LZW", len(lzw_delta)))

    # LZ77
    lz77 = lz77_compress(gray)
    results.append(("LZ77", len(lz77)))

    # LZSS
    lzss = lzss_compress(gray)
    results.append(("LZSS", len(lzss)))

    # Deflate
    deflate = zlib.compress(gray)
    results.append(("Deflate", len(deflate)))

    # Brotli
    brot = brotli.compress(gray)
    results.append(("Brotli", len(brot)))

    # Zstandard
    cctx = zstd.ZstdCompressor()
    zstd_out = cctx.compress(gray)
    results.append(("Zstandard", len(zstd_out)))

    print(f"📊 Compression Results for {image_path}:")
    print("{:<25} {:>10}".format("Method", "Size (Bytes)"))
    print("-" * 40)
    for name, size in results:
        print(f"{name:<25} {size:>10}")
    return results
