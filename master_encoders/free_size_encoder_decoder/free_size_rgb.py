import numpy as np
from PIL import Image
import brotli
import struct
from io import BytesIO # Keep although not strictly used in the main flow anymore
from collections import Counter # Keep although not strictly used in the main flow anymore
import heapq # Keep although not strictly used in the main flow anymore
import os
import math # Keep although not strictly used in the main flow anymore

### --- Utility Functions --- ###
def calculate_image_variance(img_arr):
    """Calculate variance of the image array (flattened for simplicity)"""
    # Calculate variance across all channels and pixels
    return np.var(img_arr.astype(np.float32)) # Use float32 to avoid overflow/precision issues

def delta_encode_rgb(img_array):
    """Delta encoding for RGB arrays (flattens HxWx3 to Nx3)"""
    # Reshape to N x 3
    flat_img_array = img_array.reshape(-1, 3)
    delta = np.zeros_like(flat_img_array, dtype=np.int16) # Use int16 for delta
    delta[0] = flat_img_array[0]
    delta[1:] = flat_img_array[1:].astype(np.int16) - flat_img_array[:-1].astype(np.int16)

    # Map delta values (range approx -255 to +255) to 0-510 for compression?
    # Brotli works directly on bytes. Let's encode deltas as byte pairs or similar
    # Simple approach: store delta as signed 16-bit integers, then convert to bytes
    # A better approach for byte-stream compression is to map delta to a positive range (e.g., add 255)
    # Let's map delta range [-255, 255] to [0, 510] by adding 255
    mapped_delta = delta.astype(np.int16) + 255 # int16 ensures space for 255 + 255
    return mapped_delta.astype(np.uint16) # Store as uint16

def delta_decode_rgb(delta_array):
    """Delta decoding for RGB arrays (input is mapped uint16, decode to uint8)"""
    # Map back from [0, 510] to [-255, 255] by subtracting 255
    delta = delta_array.astype(np.int16) - 255

    decoded = np.zeros_like(delta, dtype=np.int16) # Use int16 for accumulation
    decoded[0] = delta[0] + 0 # The first element was stored directly
    for i in range(1, len(delta)):
        decoded[i] = decoded[i-1] + delta[i]

    # Clamp values back to the 0-255 range
    decoded = np.clip(decoded, 0, 255)
    return decoded.astype(np.uint8) # Return as uint8

def bitpack_rgb(img_array):
    """Pack RGB values into 12 bits (4 bits per channel, lossy)"""
    # img_array is expected to be N x 3
    packed_bytes = bytearray()
    for pixel in img_array.reshape(-1, 3):
        r, g, b = pixel
        # Take the top 4 bits of R, G, B
        # Pack (R[7:4] G[7:4]) into byte 1
        # Pack (B[7:4] 0000) into byte 2
        packed_bytes.append(((r & 0xF0)) | ((g & 0xF0) >> 4))
        packed_bytes.append((b & 0xF0))
    return bytes(packed_bytes)

def bitunpack_rgb(packed_bytes):
    """Unpack 12-bit RGB values back to 24-bit"""
    pixels = []
    # Packed bytes are in pairs: byte1 (R[7:4] G[7:4]), byte2 (B[7:4] 0000)
    for i in range(0, len(packed_bytes), 2):
        byte1, byte2 = packed_bytes[i], packed_bytes[i + 1]
        # Restore the top 4 bits to 8 bits by shifting and ORing (effectively multiplying by 16)
        r = (byte1 & 0xF0)
        g = ((byte1 & 0x0F) << 4)
        b = (byte2 & 0xF0)
        pixels.append((r, g, b))
    return np.array(pixels, dtype=np.uint8)

# adaptive_chunking is not used in the current logic, so we remove it.

### --- Compression Strategies --- ###
# We will remove the Huffman strategy and rely on Brotli variants.

def compress_delta_brotli(img_arr):
    """Delta encoding + Brotli compression"""
    # Delta encode the flattened array
    delta_arr = delta_encode_rgb(img_arr.reshape(-1, 3))
    # Convert uint16 delta array to bytes
    delta_bytes = delta_arr.tobytes()
    return brotli.compress(delta_bytes)

def compress_bitpack_brotli(img_arr):
    """Bitpacking + Brotli compression (Lossy)"""
    # Bitpack the flattened array
    packed_bytes = bitpack_rgb(img_arr.reshape(-1, 3))
    return brotli.compress(packed_bytes)

def compress_brotli(img_arr):
    """Direct Brotli compression of raw pixel data"""
    return brotli.compress(img_arr.tobytes())

### --- Encoder --- ###
def encode_lix_rgb(image_path):
    """Encode RGB image to LIX format"""
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    height, width = img.height, img.width
    variance = calculate_image_variance(img_arr)

    # Choose compression method based on image characteristics and effectiveness
    # Adjusting variance thresholds and methods
    # Method 1: Delta + Brotli (Best for low variance / smooth images)
    # Method 2: Bitpack + Brotli (Lossy, good for images where 12-bit color is acceptable)
    # Method 3: Direct Brotli (General purpose fallback)

    method = 0 # Placeholder
    compressed_data = None

    # Using a threshold of 500 for "low variance" as a starting point
    # Using a threshold of 1500 for "medium variance" before falling back to general brotli
    # These thresholds may need tuning based on a diverse image dataset
    if variance < 500:
        method = 1
        print(f"Using Method 1 (Delta + Brotli) - Variance: {variance:.2f}")
        compressed_data = compress_delta_brotli(img_arr)
    elif variance < 1500:
        method = 2
        print(f"Using Method 2 (Bitpack + Brotli) - Variance: {variance:.2f}")
        compressed_data = compress_bitpack_brotli(img_arr)
    else:
        method = 3
        print(f"Using Method 3 (Direct Brotli) - Variance: {variance:.2f}")
        compressed_data = compress_brotli(img_arr)

    # Prepare header and assemble final data
    # Format: Magic (4s) + Width (H) + Height (H) + Type (B) + Method (B)
    header = b'LIXF' + struct.pack(">HHBB", width, height, 0x03, method)

    # For all new methods, the compressed data follows directly after the header
    return header + compressed_data

### --- Decoder --- ###
def decode_lix_rgb(lix_data): # Removed quality parameter as it's for JPEG saving
    """Decode LIX format back to RGB image"""
    # Header size is 4 (Magic) + 2 (Width) + 2 (Height) + 1 (Type) + 1 (Method) = 10 bytes
    header = lix_data[:10]
    magic, width, height, img_type, method = struct.unpack(">4sHHBB", header)
    assert magic == b'LIXF', "Invalid .lix file magic number"
    assert img_type == 0x03, "Invalid .lix image type (not RGB)"

    data = lix_data[10:] # The rest of the data is compressed payload

    img_arr = None

    if method == 1:
        # Delta + Brotli
        print("Decoding Method 1 (Delta + Brotli)")
        delta_bytes = brotli.decompress(data)
        # Convert bytes back to uint16 delta array
        delta_arr = np.frombuffer(delta_bytes, dtype=np.uint16)
        # Reshape delta array to N x 3 for decoding
        delta_arr = delta_arr.reshape(-1, 3)
        # Delta decode and reshape to H x W x 3
        img_arr = delta_decode_rgb(delta_arr).reshape((height, width, 3))

    elif method == 2:
        # Bitpack + Brotli
        print("Decoding Method 2 (Bitpack + Brotli)")
        packed_bytes = brotli.decompress(data)
        # Bit unpack and reshape to H x W x 3
        img_arr = bitunpack_rgb(packed_bytes).reshape((height, width, 3))

    elif method == 3:
        # Direct Brotli
        print("Decoding Method 3 (Direct Brotli)")
        decompressed_bytes = brotli.decompress(data)
        # Convert bytes to numpy array and reshape to H x W x 3
        img_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape((height, width, 3))

    else:
        raise ValueError(f"Unsupported compression method: {method}")

    # Ensure the decoded array has the correct shape and dtype
    if img_arr is None or img_arr.shape != (height, width, 3) or img_arr.dtype != np.uint8:
         raise RuntimeError("Decoding failed: Output array has incorrect shape or dtype.")

    return Image.fromarray(img_arr, 'RGB')

def save_decoded_image(image_path, output_path, save_intermediate=False, quality=85):
    """
    Encodes, decodes, and saves an image while maintaining JPEG format.

    Args:
        image_path (str): Path to input image
        output_path (str): Path to save decoded image (will be saved as JPEG)
        save_intermediate (bool): Save .lix file if True
        quality (int): JPEG quality for the *output* image (1-100)
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Ensure intermediate directory exists if needed
        lix_path = None
        if save_intermediate:
            lix_dir = "master_encoders/free_size_encoder_decoder/output"
            if not os.path.exists(lix_dir):
                os.makedirs(lix_dir)
                print(f"Created intermediate directory: {lix_dir}")
            lix_path = os.path.join(lix_dir, os.path.basename(image_path) + ".lix")

        print(f"Encoding image: {image_path}")
        # Encode the image
        lix_data = encode_lix_rgb(image_path)

        # Save intermediate if requested
        if save_intermediate and lix_path:
            with open(lix_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved .lix file: {lix_path} ({len(lix_data):,} bytes)")

        print(f"Decoding .lix data ({len(lix_data):,} bytes)")
        # Decode the image
        # The 'quality' parameter is only relevant when saving the final JPEG,
        # not during the LIX decoding itself.
        decoded_img = decode_lix_rgb(lix_data)

        # Save as JPEG with specified quality
        print(f"Saving decoded image as JPEG: {output_path} (Quality: {quality})")
        decoded_img.save(output_path, "JPEG", quality=quality)

        # Calculate statistics
        orig_size = os.path.getsize(image_path)
        compressed_size = len(lix_data)
        decoded_size = os.path.getsize(output_path) # Size of the final JPEG output

        ratio_lix_orig = orig_size / compressed_size if compressed_size > 0 else float('inf')

        print("\n--- Compression Stats ---")
        print(f"Original Image (JPEG): {os.path.basename(image_path)}")
        print(f"✅ Original Size: {orig_size:,} bytes")
        print(f"✅ Compressed (.lix) Size: {compressed_size:,} bytes")
        print(f"✅ Compression Ratio (Orig JPEG / .lix): {ratio_lix_orig:.2f}x")
        print(f"✅ Decoded Image (JPEG Output): {os.path.basename(output_path)}")
        print(f"✅ Decoded JPEG Size: {decoded_size:,} bytes (Quality: {quality})")
        # Note: Comparing decoded JPEG size to original JPEG size is a lossy-to-lossy comparison
        # The true compression is reflected in the .lix size vs. original uncompressed size,
        # but comparing to the input JPEG size is what the user requested.

        print(f"✅ Saved decoded image to: {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {image_path}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


# Example usage:
# Ensure the input image file exists at the specified path.
# Ensure the output directory exists or the function can create it.
input_image_path = "master_encoders/free_size_encoder_decoder/rgb4.jpg"
output_image_path = "master_encoders/free_size_encoder_decoder/output/output_image_rgb.jpg"

save_decoded_image(input_image_path, output_image_path, save_intermediate=True, quality=100)

# You can test with different images and quality settings.
# For example:
# save_decoded_image("path/to/another/image.png", "output/another_output.jpg", save_intermediate=True, quality=90)