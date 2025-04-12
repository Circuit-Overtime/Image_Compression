from PIL import Image
import numpy as np
import os

def pack_bw_1bit_sbw(image_array):
    flat = image_array.flatten()
    packed = []

    for i in range(0, len(flat), 8):
        byte = 0
        for j in range(8):
            if i + j < len(flat):
                byte |= (flat[i + j] & 1) << (7 - j)
        packed.append(byte)

    return bytes(packed)

def unpack_bw_1bit_sbw(packed_data, shape):  # Fixed function name for consistency
    total_pixels = shape[0] * shape[1]
    flat = []

    for byte in packed_data:
        for i in range(7, -1, -1):
            flat.append((byte >> i) & 1)
            if len(flat) == total_pixels:
                break

    return np.array(flat, dtype=np.uint8).reshape(shape)

def rle_encode_sbw(data):
    encoded = []
    prev = data[0]
    count = 1

    for byte in data[1:]:
        if byte == prev and count < 255:
            count += 1
        else:
            encoded.extend([count, prev])
            prev = byte
            count = 1

    encoded.extend([count, prev])
    return bytes(encoded)

def rle_decode_sbw(encoded):
    decoded = []
    for i in range(0, len(encoded), 2):
        count = encoded[i]
        value = encoded[i + 1]
        decoded.extend([value] * count)
    return bytes(decoded)

def decode_bw_lix_sbw(packed_data, shape):  # takes lix data (bytes) as input
    header = packed_data[:4]
    if header[:3] != b'BW1':
        raise ValueError("Not a valid BW1 LIX file.")

    mode = header[3]  # 1 = bit-packed only, 0 = RLE + packed
    data = packed_data[4:]

    if mode == 1:
        packed = data
    elif mode == 0:
        packed = rle_decode_sbw(data)
    else:
        raise ValueError("Unknown compression mode.")

    img_array = unpack_bw_1bit_sbw(packed, shape) * 255
    return Image.fromarray(img_array.astype(np.uint8), mode="L")

def encode_bw_lix_sbw(image): # takes a pillow image as input
    image_array = (np.array(image) > 0).astype(np.uint8)  # Ensure 0 and 1 explicitly
    packed = pack_bw_1bit_sbw(image_array)
    rle_packed = rle_encode_sbw(packed)
    best_data = packed if len(packed) < len(rle_packed) else rle_packed
    mode = 1 if best_data == packed else 0
    header = b'BW1' + bytes([mode])
    return header + best_data


def save_decoded_image(image_path, output_path, save_intermediate=False):
    """
    Encodes, decodes, and saves a black and white image using SBW compression.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the decoded image.
        save_intermediate (bool): If True, saves the .lix file.
    """
    try:
        # Load the image
        img = Image.open(image_path).resize((128, 128)).convert("1")

        # Encode the image
        lix_data = encode_bw_lix_sbw(img)
        shape = (img.height, img.width)

        # Save intermediate .lix file if requested
        lix_file_path = os.path.splitext(output_path)[0] + ".lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as {lix_file_path}, size: {len(lix_data)} bytes")
        else:
            print("✅ .lix file not saved (save_intermediate=False)")


        # Decode the image
        decoded_img = decode_bw_lix_sbw(lix_data, shape) # pass the lix data and size to the decoder

        # Save the decoded image
        decoded_img.save(output_path)
        print(f"✅ Decoded image saved as {output_path}")
        # Compare original and output image sizes
        original_size = os.path.getsize(image_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = original_size / output_size

        print(f"✅ Original image size: {original_size} bytes")
        print(f"✅ Compressed size: {output_size} bytes")
        print(f"✅ Compression ratio: {compression_ratio:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
