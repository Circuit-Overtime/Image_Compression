from PIL import Image
import numpy as np
from master_encoder_gray import save_decoded_image as save_decoded_image_gray
from master_encoder_rgb import save_decoded_image as save_decoded_image_rgb
from master_encoder_sbw import save_decoded_image as save_decoded_image_bw
from master_encoder_rgba import save_decoded_image as save_decoded_image_rgba

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



input_image_path = "sp_img/exp.jpg"

type = classify_image_type(input_image_path)
print(type)
if(type == 'BW'):
    save_decoded_image_bw(input_image_path, "enc_op/decoded_bw_output.jpg", save_intermediate=True)
    print("BW image has been saved!")
elif(type == 'Grayscale'):
    save_decoded_image_gray(input_image_path, "enc_op/decoded_gray_output.jpg", save_intermediate=True)
    print("Grayscale image has been saved!")
elif(type == 'RGBA'):
    save_decoded_image_rgba(input_image_path, "enc_op/decoded_rgba_output.png", save_intermediate=True)
    print("RGBA image has been saved!")
elif(type == 'RGB'):
    save_decoded_image_rgb(input_image_path, "enc_op/decoded_rgb_output.jpg", save_intermediate=True)
    print("RGB image has been saved!")
else:
    print("Unsupported image type")
