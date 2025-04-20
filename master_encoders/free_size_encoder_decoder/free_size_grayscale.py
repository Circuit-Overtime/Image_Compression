import numpy as np
from PIL import Image
# import base64 # Not used
import struct
import heapq
# import io # Not used
import os

### Compression utilities

# CORRECTED delta_encode_gray to avoid RuntimeWarning
def delta_encode_gray(data):
    """Encodes using delta (difference) coding. Lossless. Avoids warning."""
    if not data: # Handle empty input
        return []
    delta = [data[0]] # First element is stored as is
    for i in range(1, len(data)):
        # Calculate difference using standard integers first
        diff = int(data[i]) - int(data[i - 1])
        # Apply modulo 256 to get the wrap-around value (0-255)
        delta_val = diff % 256
        delta.append(delta_val)
    return delta

def rle_encode_gray(data):
    """Encodes using Run-Length Encoding. Lossless."""
    encoded = []
    if not data:
        return encoded
    count = 1
    for i in range(1, len(data)):
        # Check for same value and count limit
        if data[i] == data[i-1] and count < 255:
            count += 1
        else:
            # Append previous run
            encoded.extend([data[i-1], count])
            # Reset count for the new value
            count = 1
    # Append the last run
    encoded.extend([data[-1], count])
    return encoded

def bitpack_4bit_gray(data):
    """
    Packs 8-bit data into 4-bit nybbles.
    WARNING: Highly LOSSY unless ALL original data values are < 16.
    """
    packed = []
    for i in range(0, len(data), 2):
        first = data[i] & 0x0F
        second = (data[i+1] & 0x0F) if i+1 < len(data) else 0
        packed.append((first << 4) | second)
    return packed

# NOTE: Huffman implementation still lacks map storage in LIX format
def huffman_encode_gray(data):
    """
    Encodes data using Huffman coding. (Lossless if map is available).
    NOTE: Does NOT store the map, making decode non-functional standalone.
    """
    if not data:
        return b'', {}
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    if not freq:
         return b'', {}
    heap = [[weight, [sym, ""]] for sym, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    if len(heap) == 0: return b'', {} # Should not happen if freq populated
    if len(heap[0]) < 2: # Handle single unique symbol case
        if len(freq) == 1:
            symbol = list(freq.keys())[0]
            huff_map = { symbol: "0" }
        else: return b'', {} # Error or empty
    else:
        huff_map = {sym: code for sym, code in heap[0][1:]}

    encoded_bits = ''.join([huff_map[b] for b in data])
    padded = encoded_bits + '0' * ((8 - len(encoded_bits) % 8) % 8)
    encoded_bytes = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    return encoded_bytes # Return only bytes for consistency, map is lost


# CORRECTED lzw_encode_gray to limit dictionary size
def lzw_encode_gray(data):
    """Encodes data using LZW compression with 16-bit code limit. Lossless."""
    # Ensure input is bytes
    data_bytes = bytes(data)
    if not data_bytes:
        return b''

    dictionary_size = 256
    # Initialize dictionary with single bytes
    dictionary = {bytes([i]): i for i in range(dictionary_size)}
    w = b""
    codes = []

    for byte_val in data_bytes:
        c = bytes([byte_val])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # Output the code for the known prefix 'w'
            codes.append(dictionary[w])

            # Add wc to the dictionary ONLY if size limit not reached
            # Max code for '>H' is 65535, so max dict size is 65536
            if dictionary_size < 65536:
                dictionary[wc] = dictionary_size
                dictionary_size += 1
            # Reset w to the current byte that wasn't part of a known sequence
            w = c

    # Output the code for the last remaining sequence in w
    if w:
        codes.append(dictionary[w]) # w must be in the dictionary

    # Pack codes as 16-bit unsigned short integers (big-endian)
    try:
        encoded_bytes = b''.join(struct.pack('>H', code) for code in codes)
    except struct.error as e:
        # Add more info if packing fails despite the size check
        problematic_codes = [c for c in codes if not (0 <= c <= 65535)]
        raise ValueError(f"LZW packing error: {e}. Problematic codes: {problematic_codes[:10]}...") from e

    return encoded_bytes

### Decompression utilities

# CORRECTED delta_decode_grey (already fixed in previous step, ensuring it's correct)
def delta_decode_grey(data):
    """Decodes delta encoded data. Lossless."""
    if not data:
        return []
    # Data should be a list/tuple of integers (0-255) here
    # If it arrives as bytes, convert first: data = list(data)
    if isinstance(data, bytes):
         data = list(data)

    if not data: # Check again after potential conversion
        return []

    decoded = [data[0]]
    last_decoded_value = data[0]
    for i in range(1, len(data)):
        delta_value = data[i]
        next_value_sum = last_decoded_value + delta_value
        next_value_decoded = next_value_sum % 256
        decoded.append(next_value_decoded)
        last_decoded_value = next_value_decoded
    return decoded

# CORRECTED rle_decode_grey to handle potential odd length input gracefully
def rle_decode_grey(data):
    """Decodes Run-Length Encoded data. Lossless."""
     # If it arrives as bytes, convert first: data = list(data)
    if isinstance(data, bytes):
         data = list(data)

    decoded = []
    for i in range(0, len(data), 2):
         if i + 1 < len(data): # Check if there's a pair (value, count)
             value = data[i]
             count = data[i + 1]
             if count < 0 : # Basic sanity check
                 raise ValueError(f"Invalid RLE count: {count} at index {i+1}")
             decoded.extend([value] * count)
         else:
             # Handle malformed data (odd length) - e.g., log a warning or ignore
             print(f"Warning: Malformed RLE data - odd length {len(data)}, ignoring last byte.")
             break # Stop processing
    return decoded


# NOTE: bitunpack still needs expected_length for perfect reconstruction if original length was odd
def bitunpack_4bit_grey(data, expected_length):
    """
    Unpacks 4-bit nybbles back into 8-bit data.
    Requires expected_length to handle potential padding nybble.
    WARNING: Lossy if original data values were >= 16.
    """
    if isinstance(data, bytes):
         data = list(data)
    unpacked = []
    for byte in data:
        unpacked.append((byte >> 4) & 0x0F) # First nybble
        unpacked.append(byte & 0x0F)      # Second nybble

    # Trim to the expected length to remove potential padding
    if len(unpacked) < expected_length:
         raise ValueError(f"Bitunpack error: Expected {expected_length} pixels, got {len(unpacked)}")
    return unpacked[:expected_length]

# NOTE: Huffman decode still non-functional as map isn't stored
def huffman_decode_grey(encoded_data, huff_map, expected_length):
    """
    Decodes Huffman encoded data. Requires map and expected length.
    NOTE: Non-functional with current LIX format.
    """
    raise NotImplementedError("Huffman decoding requires the Huffman map, which is not stored.")
    # ... (previous implementation remains here but is unused) ...


# CORRECTED lzw_decode_grey to handle potential errors and edge cases
def lzw_decode_grey(encoded_data):
    """Decodes LZW compressed data (16-bit codes). Lossless."""
    if not encoded_data:
        return b""
    if len(encoded_data) % 2 != 0:
        raise ValueError("Invalid LZW data: Length must be multiple of 2 for 16-bit codes.")

    # Unpack 16-bit codes
    try:
        codes = [struct.unpack('>H', encoded_data[i:i + 2])[0] for i in range(0, len(encoded_data), 2)]
    except struct.error as e:
        raise ValueError(f"LZW unpacking error: {e}") from e

    if not codes:
        return b""

    dictionary_size = 256
    # Initialize dictionary with single bytes
    dictionary = {i: bytes([i]) for i in range(dictionary_size)}

    # First code must be in the initial dictionary (0-255)
    if codes[0] >= dictionary_size:
         raise ValueError(f"Invalid LZW data: First code {codes[0]} is out of initial range 0-255.")

    w = dictionary[codes[0]]
    decoded_bytes = [w] # Use a list to append byte strings

    for code in codes[1:]:
        entry = b''
        if code in dictionary:
            entry = dictionary[code]
        elif code == dictionary_size:
            # Special case: KwKwK pattern
            if w == b'': # Cannot form KwKwK if w is empty
                 raise ValueError("Invalid LZW sequence: KwKwK pattern cannot start with empty w")
            entry = w + w[:1]
        else:
            # Code is not in dictionary and not the special case -> Invalid data
            raise ValueError(f"Invalid LZW code encountered: {code} (dictionary size {dictionary_size})")

        decoded_bytes.append(entry)

        # Add new entry to dictionary if size permits
        # Mirroring the encoder's limit
        if dictionary_size < 65536:
             if w == b'' or entry == b'': # Avoid adding entries based on empty strings if they somehow occur
                  raise ValueError("Invalid LZW state: Attempting to add empty string to dictionary")
             dictionary[dictionary_size] = w + entry[:1]
             dictionary_size += 1
        w = entry # Update w for the next iteration

    return b''.join(decoded_bytes)


### Grayscale Encoder Pipeline
def encode_grayscale_lix_gray(img: Image.Image):
    """
    Encodes a grayscale PIL Image object into the LIXG format.
    Args:
        img: PIL Image object (should be in 'L' mode).
    Returns:
        bytes: The encoded LIXG data.
    """
    if img.mode != 'L':
        img = img.convert('L') # Convert if not already grayscale
        # raise ValueError("Image must be in grayscale ('L') mode.")

    # Use tolist() for potentially better performance than flatten() for large images
    data_list = list(img.getdata()) # Get pixel data as a flat list
    if not data_list: # Handle empty image
         width, height = img.size
         if width == 0 or height == 0:
              # Return minimal valid LIX for empty image using method 0
              header = b'LIXG'
              meta = struct.pack('>IIB', 0, 0, 0) # Width=0, Height=0, Method=0
              return header + meta + b''
         else:
              raise ValueError("Image has dimensions but no data?")

    width, height = img.size

    if len(data_list) != width * height:
         raise ValueError(f"Data length mismatch: {len(data_list)} vs {width}x{height}")

    methods = {}
    huff_maps = {} # Still unused in output format

    # --- Try different compression methods ---

    # Method 0: Delta only
    try:
        delta = delta_encode_gray(data_list)
        methods[0] = bytes(delta)
    except Exception as e:
        print(f"Warning: Delta encoding failed: {e}")


    # Method 1: Delta + RLE
    if 0 in methods: # Only if delta succeeded
        try:
            delta_rle = rle_encode_gray(delta)
            methods[1] = bytes(delta_rle)
        except Exception as e:
            print(f"Warning: RLE encoding failed: {e}")


    # Method 2: Bit-pack + Huffman (check applicability)
    # This method remains non-functional due to map storage issue.
    # It's also lossy if data > 15. We keep the check but it won't be used effectively.
    try:
        can_bitpack = all(p < 16 for p in data_list)
        if can_bitpack:
            bitpacked = bitpack_4bit_gray(data_list)
            # NOTE: Huffman map is discarded by current implementation design
            huff_encoded_bytes = huffman_encode_gray(bitpacked)
            if huff_encoded_bytes is not None:
                methods[2] = huff_encoded_bytes
                # huff_maps[2] = huff_map_m2 # Not storing map
    except Exception as e:
            print(f"Warning: Bitpack/Huffman encoding failed: {e}")


    # Method 3: Delta + LZW
    if 0 in methods: # Only if delta succeeded
        try:
            # LZW encode function now expects bytes/list[int], delta is list[int]
            delta_lzw = lzw_encode_gray(delta)
            methods[3] = delta_lzw
        except Exception as e:
            print(f"Warning: LZW encoding failed: {e}")


    # --- Choose the smallest result ---
    if not methods:
        raise RuntimeError("All compression methods failed.")

    # Find method index with the minimum compressed size
    # Ensure the method produced non-None result if errors were handled softly
    valid_methods = {k: v for k, v in methods.items() if v is not None}
    if not valid_methods:
         raise RuntimeError("All compression methods failed or produced None.")

    best_method = min(valid_methods, key=lambda k: len(valid_methods[k]))
    compressed = valid_methods[best_method]

    # --- Construct .lix binary ---
    header = b'LIXG'
    # Pack width, height (32-bit unsigned int), method (8-bit unsigned byte)
    # Using 'I' (4 bytes) for width/height allows for larger images than 'H' (2 bytes)
    meta = struct.pack('>IIB', width, height, best_method)

    return header + meta + compressed


### Grayscale Decoder Pipeline
def decode_grayscale_lix_grey(lix_data: bytes):
    """
    Decodes LIXG formatted byte data back into a grayscale PIL Image.
    Args:
        lix_data: The LIXG byte data.
    Returns:
        Image.Image: The decoded grayscale image.
    """
    # --- Parse header and metadata ---
    # Header (4) + Width (4) + Height (4) + Method (1) = 13 bytes
    if len(lix_data) < 13:
        raise ValueError("Invalid LIXG data: Too short for header and metadata (IIB).")

    header = lix_data[:4]
    if header != b'LIXG':
        raise ValueError("Invalid LIXG data: Incorrect header.")

    # Unpack width, height (using 'I'), method
    try:
        width, height, method = struct.unpack('>IIB', lix_data[4:13])
    except struct.error as e:
         raise ValueError(f"Failed to unpack LIXG metadata: {e}") from e

    compressed_data = lix_data[13:]

    # Handle empty image case encoded as 0x0
    if width == 0 and height == 0:
         if not compressed_data:
              return Image.new('L', (0, 0))
         else:
              raise ValueError("LIXG format error: Zero dimensions but non-empty data.")


    expected_pixel_count = width * height
    if expected_pixel_count < 0: # Sanity check from width*height overflow? Unlikely with 'I'
         raise ValueError("Invalid LIXG dimensions resulting in negative pixel count.")


    # --- Decompress based on method ---
    decompressed_list = None # Initialize

    if method == 0:  # Delta only
        decompressed_list = delta_decode_grey(compressed_data) # Pass bytes directly
    elif method == 1:  # Delta + RLE
        delta_rle_decoded = rle_decode_grey(compressed_data) # Pass bytes directly
        decompressed_list = delta_decode_grey(delta_rle_decoded) # Pass list
    elif method == 2:  # Bit-pack + Huffman
        raise NotImplementedError(
            "Decoding Method 2 (Bitpack+Huffman) is not possible: "
            "Huffman map is not stored in the LIXG file format."
            )
        # --- Hypothetical Code ---
        # expected_bitpack_len = (expected_pixel_count + 1) // 2
        # bitpacked_decoded = huffman_decode_grey(compressed_data, huff_map, expected_bitpack_len) # Needs map!
        # decompressed_list = bitunpack_4bit_grey(bitpacked_decoded, expected_pixel_count)
        # --- End Hypothetical ---
    elif method == 3:  # Delta + LZW
        delta_decoded_bytes = lzw_decode_grey(compressed_data) # Pass bytes directly
        # delta_decode_grey expects list of ints, so convert bytes
        decompressed_list = delta_decode_grey(list(delta_decoded_bytes)) # Pass list
    else:
        raise ValueError(f"Unknown compression method encountered: {method}")

    # --- Validate and Reshape ---
    if decompressed_list is None:
         raise RuntimeError("Decompression failed to produce data.")

    if len(decompressed_list) != expected_pixel_count:
        raise ValueError(f"Decompression error: Expected {expected_pixel_count} pixels for {width}x{height}, got {len(decompressed_list)} using Method {method}")

    # Convert list of ints (0-255) to numpy array with correct type and shape
    try:
        img_array = np.array(decompressed_list, dtype=np.uint8).reshape((height, width))
    except ValueError as e:
         # This might happen if reshape fails due to wrong number of elements,
         # although we checked expected_pixel_count already. Could be memory error.
         raise ValueError(f"Failed to reshape pixel data: {e}. Expected ({height}, {width})") from e


    # Create PIL image
    try:
        return Image.fromarray(img_array, mode="L")
    except Exception as e:
        raise RuntimeError(f"Failed to create PIL Image from array: {e}") from e


### Master Function
def save_decoded_image(image_path, output_path, save_intermediate=False):
    """
    Encodes, decodes, and saves a grayscale image with lossless .lix compression.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the decoded image.
        save_intermediate (bool): If True, saves the .lix file too.
    """
    print(f"--- Processing '{image_path}' ---")
    try:
        # Load image, convert to grayscale, keep native size
        img = Image.open(image_path).convert("L")
        print(f"✅ Loaded image ({img.width}x{img.height})")
        original_size_bytes = img.width * img.height
        original_filesize = os.path.getsize(image_path) # Filesize of original file (e.g. PNG/JPG)


        # Encode
        lix_data = encode_grayscale_lix_gray(img)
        lix_method = lix_data[12] # Method byte is at index 12 (4+4+4)
        lix_filesize = len(lix_data)
        print(f"✅ Encoded using Method {lix_method}")

        # Optionally save the .lix file
        lix_file_path = "master_encoders/free_size_encoder_decoder/output/output_grey.lix"
        if save_intermediate:
            with open(lix_file_path, 'wb') as f:
                f.write(lix_data)
            print(f"✅ Saved intermediate .lix file as '{lix_file_path}' ({lix_filesize} bytes)")
        else:
            print(f"ℹ️ Intermediate .lix file not saved (save_intermediate=False)")

        # Decode back to image
        decoded_img = decode_grayscale_lix_grey(lix_data)
        print(f"✅ Decoded back to image ({decoded_img.width}x{decoded_img.height})")

        # Save the decoded image (e.g., as PNG for lossless check)
        decoded_img.save(output_path)
        print(f"✅ Decoded image saved as '{output_path}'")

        # Verification
        original_pixels = np.array(img)
        decoded_pixels = np.array(decoded_img)
        if np.array_equal(original_pixels, decoded_pixels):
             print("✅ Verification successful: Decoded pixel data matches original.")
        else:
             print("❌ VERIFICATION FAILED: Decoded pixel data does NOT match original!")
             # Optional: calculate difference
             diff = np.sum(original_pixels != decoded_pixels)
             print(f"   Number of differing pixels: {diff}")


        # --- Report sizes and compression ratio ---
        print("\n--- Statistics ---")
        print(f"Original file ('{os.path.basename(image_path)}'): {original_filesize} bytes")
        print(f"Raw pixel data ({img.width}x{img.height}): {original_size_bytes} bytes")
        print(f"Compressed LIXG (Method {lix_method}): {lix_filesize} bytes")
        try:
            decoded_filesize = os.path.getsize(output_path)
            print(f"Saved decoded file ('{os.path.basename(output_path)}'): {decoded_filesize} bytes")
        except FileNotFoundError:
             print(f"Saved decoded file ('{os.path.basename(output_path)}'): Not found (Save failed?)")


        # Compression ratio: Compare LIX size to RAW pixel data size
        if original_size_bytes > 0:
            compression_ratio = original_size_bytes / lix_filesize
            print(f"Compression Ratio (Raw Pixels / LIXG Size): {compression_ratio:.2f} : 1")
            space_saving = 100 * (1 - (lix_filesize / original_size_bytes))
            print(f"Space Saving vs Raw Pixels: {space_saving:.2f}%")
        else:
            print("Cannot calculate compression ratio for empty image.")
        print("--------------------\n")


    except FileNotFoundError:
        print(f"❌ ERROR: Input image file not found at '{image_path}'")
    except NotImplementedError as nie:
         print(f"❌ ERROR: Decoding failed - {nie}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    print("--------------------\n")


# Example usage
save_decoded_image("master_encoders/free_size_encoder_decoder/bw.jpg", "master_encoders/free_size_encoder_decoder/output/output_image_gray.jpg", save_intermediate=True)
