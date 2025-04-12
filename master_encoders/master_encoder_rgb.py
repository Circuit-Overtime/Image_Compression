import numpy as np
from PIL import Image
import brotli
import struct
from io import BytesIO
from collections import Counter
import heapq
import os


### --- Utility Functions --- ###
def calculate_image_variance(img_arr):
    return np.var(img_arr)

def delta_encode_rgb(img_array):
    delta = np.zeros_like(img_array)
    delta[0] = img_array[0]
    delta[1:] = (img_array[1:].astype(np.int16) - img_array[:-1].astype(np.int16)) % 256
    return delta.astype(np.uint8)

def delta_decode_rgb(delta_array):
    decoded = np.zeros_like(delta_array, dtype=np.uint8)
    decoded[0] = delta_array[0]
    for i in range(1, len(delta_array)):
        decoded[i] = (decoded[i-1].astype(np.int16) + delta_array[i]) % 256
    return decoded

def bitpack_rgb(img_array):
    packed = bytearray()
    for r, g, b in img_array:
        packed.append(((r >> 4) << 4) | (g >> 4))
        packed.append((b >> 4) << 4)
    return bytes(packed)

def bitunpack_rgb(packed_bytes):
    pixels = []
    for i in range(0, len(packed_bytes), 2):
        byte1, byte2 = packed_bytes[i], packed_bytes[i + 1]
        r = (byte1 >> 4) << 4
        g = (byte1 & 0x0F) << 4
        b = (byte2 >> 4) << 4
        pixels.append((r, g, b))
    return np.array(pixels, dtype=np.uint8)


### --- Huffman --- ###
class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = self.right = None
    def __lt__(self, other): return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = [Node(k, v) for k, v in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        parent = Node(freq=left.freq + right.freq)
        parent.left, parent.right = left, right
        heapq.heappush(heap, parent)
    return heap[0]

def build_huffman_dict(node, prefix='', codebook=None):
    if codebook is None: codebook = {}
    if node.char is not None:
        codebook[node.char] = prefix
    else:
        build_huffman_dict(node.left, prefix + '0', codebook)
        build_huffman_dict(node.right, prefix + '1', codebook)
    return codebook

def compress_huffman_with_codebook(data):
    tree = build_huffman_tree(data)
    codebook = build_huffman_dict(tree)
    encoded_bits = ''.join(codebook[byte] for byte in data)
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += '0' * padding
    compressed = bytearray(int(encoded_bits[i:i+8], 2) for i in range(0, len(encoded_bits), 8))

    # Serialize codebook
    cb = bytearray()
    cb.append(len(codebook))
    for byte, code in codebook.items():
        cb.append(byte)
        cb.append(len(code))
        cb.extend(int(code, 2).to_bytes((len(code) + 7) // 8, 'big'))
    return bytes(cb)

def build_huffman_tree_from_dict(codebook):
    root = Node()
    for byte, code in codebook.items():
        node = root
        for bit in code:
            if bit == '0':
                node.left = node.left or Node()
                node = node.left
            else:
                node.right = node.right or Node()
                node = node.right
        node.char = byte
    return root

def decompress_huffman(data, codebook, padding):
    bitstring = ''.join(f"{byte:08b}" for byte in data)
    bitstring = bitstring[:-padding] if padding else bitstring
    root = build_huffman_tree_from_dict(codebook)
    node, output = root, bytearray()
    for bit in bitstring:
        node = node.left if bit == '0' else node.right
        if node.char is not None:
            output.append(node.char)
            node = root
    return bytes(output)


### --- Compression Strategies --- ###
def compress_delta_huffman_with_codebook(img_arr):
    delta = delta_encode_rgb(img_arr)
    return compress_huffman_with_codebook(delta.flatten())

def compress_bitpack_brotli(img_arr):
    return brotli.compress(bitpack_rgb(img_arr))

def compress_delta_bitpack_brotli(img_arr):
    return brotli.compress(bitpack_rgb(delta_encode_rgb(img_arr)))

def compress_brotli(img_arr):
    return brotli.compress(img_arr.flatten().tobytes())


### --- Encoder --- ###
def encode_lix_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img).reshape(-1, 3)
    height, width = img.height, img.width
    variance = calculate_image_variance(img_arr)

    if variance < 300:
        method = 1; compressed = compress_delta_huffman_with_codebook(img_arr)
    elif variance < 800:
        method = 2; compressed = compress_bitpack_brotli(img_arr)
    elif variance < 1500:
        method = 3; compressed = compress_delta_bitpack_brotli(img_arr)
    else:
        method = 4; compressed = compress_brotli(img_arr)

    header = b'LIXF' + struct.pack(">HHBB", width, height, 0x03, method)
    return header + compressed


### --- Decoder --- ###
def decode_lix_rgb(lix_data, quality=85):  # Added quality parameter, takes lix_data
    content = lix_data

    header = content[:10]
    magic, width, height, img_type, method = struct.unpack(">4sHHBB", header)
    assert magic == b'LIXF' and img_type == 0x03, "Invalid .lix RGB file"
    data = content[10:]

    if method == 1:
        ptr = 0
        entries = data[ptr]; ptr += 1
        codebook = {}
        for _ in range(entries):
            byte = data[ptr]; ptr += 1
            length = data[ptr]; ptr += 1
            code_len = (length + 7) // 8
            code = bin(int.from_bytes(data[ptr:ptr + code_len], 'big'))[2:].zfill(length)
            codebook[byte] = code
            ptr += code_len
        padding = data[ptr]; ptr += 1
        flat = decompress_huffman(data[ptr:], codebook, padding)
        delta_arr = np.frombuffer(flat, dtype=np.uint8).reshape(-1, 3)
        img_arr = delta_decode_rgb(delta_arr)

    elif method == 2:
        img_arr = bitunpack_rgb(brotli.decompress(data))
    elif method == 3:
        delta_arr = bitunpack_rgb(brotli.decompress(data))
        img_arr = delta_decode_rgb(delta_arr)
    elif method == 4:
        img_arr = np.frombuffer(brotli.decompress(data), dtype=np.uint8).reshape(height, width, 3)
    else:
        raise ValueError("Unsupported method")

    if method != 4:
        img_arr = img_arr.reshape((height, width, 3))

    return Image.fromarray(img_arr, 'RGB')



def save_decoded_image(image_path, output_path, save_intermediate=False, quality=85):
    """
    Encodes, decodes, and saves an image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the decoded image.
        save_intermediate (bool): If True, saves the .lix file.
        quality (int): JPEG quality for the decoded image (0-100).
    """
    try:
        # Encode the image
        lix_data = encode_lix_rgb(image_path)

        # Save intermediate .lix file if requested
        lix_file_path = os.path.splitext(output_path)[0] + ".lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as {lix_file_path}, size: {len(lix_data)} bytes")
        else:
            print("✅ .lix file not saved (save_intermediate=False)")

        # Decode the image
        decoded_img = decode_lix_rgb(lix_data, quality=quality)

        # Save the decoded image
        decoded_img.save(output_path, "JPEG", quality=quality) # force jpeg and include quality
        print(f"✅ Decoded and saved to {output_path} as JPEG with quality {quality}")
        # Compare original and decoded images
        original_size = os.path.getsize(image_path)
        decoded_size = os.path.getsize(output_path)
        compression_ratio = original_size / len(lix_data)

        print(f"✅ Original Image Size: {original_size} bytes")
        print(f"✅ Compressed .lix Size: {len(lix_data)} bytes")
        print(f"✅ Decoded Image Size: {decoded_size} bytes")
        print(f"✅ Compression Ratio: {compression_ratio:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    image_path = "./sp_img/exp4.jpg" # replace with a path to your image
    decoded_path = "./enc_op/decoded_rgb.jpg"  # Save as JPG

    save_decoded_image(image_path, decoded_path, save_intermediate=True)