from PIL import Image
import numpy as np

def classify_image_type(image_path):
    img = Image.open(image_path)
    mode = img.mode

    # If already clear from mode
    if mode == '1':
        return 'BW'
    elif mode == 'L':
        arr = np.array(img)
        unique_vals = np.unique(arr)
        if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
            return 'BW'
        return 'Grayscale'
    elif mode == 'RGBA':
        return 'RGBA'
    elif mode == 'RGB':
        arr = np.array(img)
        # Check if all channels are equal (grayscale stored in RGB)
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            unique_vals = np.unique(arr[:, :, 0])
            if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
                return 'BW'
            return 'Grayscale'
        return 'RGB'
    else:
        return 'Unknown'



# Example usage:
type = classify_image_type("op_img/decoded_rgba_output.png")
print(type)
