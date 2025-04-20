from PIL import Image
import numpy as np
import os
import struct # Needed for packing/unpacking dimensions

# --- Compression/Decompression Utilities ---

def pack_bw_1bit_sbw(image_array_01):
    """Packs a flat numpy array of 0s and 1s into bytes (1 bit per pixel)."""
    flat = image_array_01.flatten()
    packed = bytearray() # Use bytearray for efficient appending

    byte_count = (len(flat) + 7) // 8 # Calculate needed bytes
    bit_index = 0
    current_byte = 0

    for pixel in flat:
        # Add the pixel (0 or 1) to the current byte at the correct bit position
        # We pack from MSB (leftmost) to LSB (rightmost) within each byte
        current_byte |= (pixel & 1) << (7 - (bit_index % 8))
        bit_index += 1

        # If we've filled a byte (or it's the last pixel), append it
        if bit_index % 8 == 0:
            packed.append(current_byte)
            current_byte = 0

    # Append the last byte if it wasn't fully filled
    if bit_index % 8 != 0:
        packed.append(current_byte)

    return bytes(packed)

def unpack_bw_1bit_sbw(packed_data, width, height):
    """Unpacks 1-bit packed data into a 0/1 numpy array of given shape."""
    total_pixels = width * height
    if total_pixels == 0:
        return np.array([], dtype=np.uint8).reshape((height, width))

    flat = np.zeros(total_pixels, dtype=np.uint8)
    pixel_index = 0

    for byte in packed_data:
        for bit_pos in range(7, -1, -1): # Iterate bits from MSB to LSB
            if pixel_index < total_pixels:
                # Extract the bit and assign it
                pixel_value = (byte >> bit_pos) & 1
                flat[pixel_index] = pixel_value
                pixel_index += 1
            else:
                # Stop if we have unpacked all required pixels
                break
        if pixel_index == total_pixels:
             break # Exit outer loop too

    # Check if we got the expected number of pixels
    if pixel_index != total_pixels:
        raise ValueError(f"Unpacking error: Expected {total_pixels} pixels, but only processed {pixel_index} from packed data.")

    return flat.reshape((height, width))

def rle_encode_sbw(byte_data):
    """Performs Run-Length Encoding on byte data. Format: [Count, Value]"""
    if not byte_data:
        return b''

    encoded = bytearray()
    prev_byte = byte_data[0]
    count = 1

    for current_byte in byte_data[1:]:
        if current_byte == prev_byte and count < 255:
            count += 1
        else:
            # Append the run: count (1 byte), value (1 byte)
            encoded.append(count)
            encoded.append(prev_byte)
            # Reset for the new run
            prev_byte = current_byte
            count = 1

    # Append the very last run
    encoded.append(count)
    encoded.append(prev_byte)
    return bytes(encoded)

def rle_decode_sbw(encoded_bytes):
    """Decodes Run-Length Encoded byte data. Expects [Count, Value] format."""
    if not encoded_bytes:
        return b''
    if len(encoded_bytes) % 2 != 0:
        raise ValueError("Invalid RLE data: Length must be even.")

    decoded = bytearray()
    for i in range(0, len(encoded_bytes), 2):
        count = encoded_bytes[i]
        value = encoded_bytes[i + 1]
        if count == 0:
             raise ValueError("Invalid RLE data: Count cannot be zero.")
        # Extend the bytearray efficiently
        decoded.extend([value] * count)
    return bytes(decoded)

# --- LIX Format Encoder/Decoder ---

def encode_bw_lix_sbw(image: Image.Image):
    """
    Encodes a 1-bit PIL Image into the BW1 LIX format.
    Selects between raw bit-packing or RLE on the packed data.
    Format: BW1[Mode][Width][Height][Data...]
            Mode: 1 byte (0=RLE+Packed, 1=Packed Only)
            Width: 4 bytes (Big-endian unsigned int)
            Height: 4 bytes (Big-endian unsigned int)
    """
    if image.mode != '1':
        image = image.convert('1') # Ensure image is 1-bit

    width, height = image.size

    # Convert PIL image to numpy array of 0s and 1s
    # Note: PIL '1' mode can store 0/255. We need 0/1 for packing.
    image_array = (np.array(image) > 0).astype(np.uint8) # True->1, False->0

    # 1. Pack the 0/1 data into bytes
    packed_1bit_data = pack_bw_1bit_sbw(image_array)

    # 2. Try RLE on the packed data
    rle_on_packed_data = rle_encode_sbw(packed_1bit_data)

    # 3. Choose the smaller representation
    best_data = packed_1bit_data
    mode = 1 # Mode 1: Packed Only
    if len(rle_on_packed_data) < len(packed_1bit_data):
        best_data = rle_on_packed_data
        mode = 0 # Mode 0: RLE applied to packed data

    # 4. Construct header: BW1 + Mode + Width + Height
    header = b'BW1' + struct.pack('>BII', mode, width, height)
    # B = 1 byte unsigned char (for mode)
    # I = 4 byte unsigned int (for width/height)
    # > = Big-endian

    return header + best_data


def decode_bw_lix_sbw(lix_data: bytes):
    """
    Decodes BW1 LIX formatted byte data back into a 1-bit PIL Image.
    Reads dimensions and mode from the header.
    """
    # --- Parse Header ---
    # Header: BW1 (3) + Mode (1) + Width (4) + Height (4) = 12 bytes
    if len(lix_data) < 12:
        raise ValueError("Invalid BW1 LIX data: Too short for header.")

    if lix_data[:3] != b'BW1':
        raise ValueError("Not a valid BW1 LIX file (Incorrect magic bytes).")

    try:
        mode, width, height = struct.unpack('>BII', lix_data[3:12])
    except struct.error as e:
        raise ValueError(f"Failed to unpack BW1 LIX metadata: {e}") from e

    compressed_payload = lix_data[12:]

    # --- Decompress Payload based on Mode ---
    packed_1bit_data = b''
    if mode == 1: # Mode 1: Data is directly the bit-packed stream
        packed_1bit_data = compressed_payload
        print("   (Decoding Mode 1: Packed Only)")
    elif mode == 0: # Mode 0: Data is RLE encoded stream of packed bits
        packed_1bit_data = rle_decode_sbw(compressed_payload)
        print("   (Decoding Mode 0: RLE + Packed)")
    else:
        raise ValueError(f"Unknown compression mode in BW1 LIX header: {mode}")

    # --- Unpack bits ---
    # unpack_bw_1bit_sbw needs the target dimensions
    image_array_01 = unpack_bw_1bit_sbw(packed_1bit_data, width, height)

    # --- Convert back to PIL Image ---
    # Multiply by 255 to get black/white suitable for 'L' mode display
    # or keep as 0/1 if saving directly as '1' mode (less common save format)
    image_array_bw = (image_array_01 * 255).astype(np.uint8)
    return Image.fromarray(image_array_bw, mode="L") # Return as L mode for easier viewing/saving


# --- Master Function ---

def save_decoded_image(image_path, output_path, save_intermediate=False):
    """
    Encodes, decodes, and saves a black and white image using SBW (BW1 LIX) compression.
    Works with arbitrary image sizes.

    Args:
        image_path (str): Path to the input image (will be converted to 1-bit).
        output_path (str): Path to save the decoded image (usually as PNG/BMP).
        save_intermediate (bool): If True, saves the intermediate .lix file.
    """
    print(f"--- Processing '{image_path}' ---")
    try:
        # Load the image and convert to 1-bit immediately
        # REMOVED the .resize() call
        img = Image.open(image_path).convert("1")
        print(f"✅ Loaded image ({img.width}x{img.height}), converted to 1-bit")
        original_filesize = os.path.getsize(image_path)

        # Encode the image
        lix_data = encode_bw_lix_sbw(img)
        lix_mode = lix_data[3] # Mode byte is at index 3
        lix_filesize = len(lix_data)
        print(f"✅ Encoded using Mode {lix_mode} (0=RLE+Packed, 1=Packed Only)")

        # Save intermediate .lix file if requested
        lix_file_path = "master_encoders/free_size_encoder_decoder/output/output__sbw_lix.lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as '{lix_file_path}' ({lix_filesize} bytes)")
        else:
            print("ℹ️ Intermediate .lix file not saved (save_intermediate=False)")

        # Decode the image
        print("✅ Decoding...")
        decoded_img = decode_bw_lix_sbw(lix_data) # Decoder now gets shape from lix_data
        print(f"✅ Decoded back to image ({decoded_img.width}x{decoded_img.height})")

        # Save the decoded image (as PNG for lossless display/check)
        decoded_img.save(output_path)
        print(f"✅ Decoded image saved as '{output_path}'")

        # Verification (Compare the 0/1 pixel data)
        original_pixels_01 = (np.array(img) > 0).astype(np.uint8)
        # Need to convert decoded image (which is L mode 0/255) back to 0/1
        decoded_pixels_01 = (np.array(decoded_img) > 0).astype(np.uint8)
        if np.array_equal(original_pixels_01, decoded_pixels_01):
             print("✅ Verification successful: Decoded pixel data matches original.")
        else:
             print("❌ VERIFICATION FAILED: Decoded pixel data does NOT match original!")
             diff = np.sum(original_pixels_01 != decoded_pixels_01)
             print(f"   Number of differing pixels: {diff}")


        # --- Report sizes and compression ratio ---
        print("\n--- Statistics ---")
        print(f"Original file ('{os.path.basename(image_path)}'): {original_filesize} bytes")
        # Calculate theoretical minimum size for 1-bit raw data
        raw_1bit_size_bytes = (img.width * img.height + 7) // 8
        print(f"Raw 1-bit pixel data ({img.width}x{img.height}): {raw_1bit_size_bytes} bytes")
        print(f"Compressed LIX (Mode {lix_mode}): {lix_filesize} bytes")
        try:
            decoded_filesize = os.path.getsize(output_path)
            print(f"Saved decoded file ('{os.path.basename(output_path)}'): {decoded_filesize} bytes")
        except FileNotFoundError:
             print(f"Saved decoded file ('{os.path.basename(output_path)}'): Not found (Save failed?)")

        # Compression ratio: Compare LIX size to RAW 1-bit pixel data size
        if raw_1bit_size_bytes > 0:
            # Ratio < 1 means RLE made it bigger (plus header overhead)
            # Ratio > 1 means RLE helped (or raw packing was used and header is small relative to data)
            compression_ratio = raw_1bit_size_bytes / (lix_filesize - 12) if lix_filesize > 12 else 0 # Compare payload to raw
            print(f"Compression Ratio (Raw 1-bit / LIX Payload Size): {compression_ratio:.2f} : 1")
            space_saving = 100 * (1 - (lix_filesize / raw_1bit_size_bytes)) if raw_1bit_size_bytes else 0
            # Negative saving means expansion
            print(f"Space Saving vs Raw 1-bit Pixels: {space_saving:.2f}%")
        else:
            print("Cannot calculate compression ratio for empty image.")
        print("--------------------\n")

    except FileNotFoundError:
        print(f"❌ ERROR: Input image file not found at '{image_path}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("--------------------\n")


save_decoded_image("master_encoders/free_size_encoder_decoder/sbw.jpg", "master_encoders/free_size_encoder_decoder/output/output_image_sbw.jpg", save_intermediate=True)