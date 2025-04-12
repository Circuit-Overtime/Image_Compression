import requests
from PIL import Image
from io import BytesIO
from rembg import remove
import os

rgba_prompts = [
    "isolated red apple on white background",
    "single coffee mug on white background",
    "modern chair on plain white background",
    "cute puppy sitting alone on white backdrop",
    "colorful parrot standing on white background",
    "a futuristic helmet on a white surface",
    "minimalist watch on a white background",
    "toy robot placed on white floor",
    "single sneaker shoe on white backdrop",
    "green succulent plant in pot on white table",
    "laptop with white background",
    "glass bottle on white surface",
    "bottle of perfume on white background",
    "slice of pizza on clean white plate",
    "colorful donut on a white table",
    "cat sleeping on white cushion",
    "vintage camera on white background",
    "a banana on white surface",
    "small drone on plain white background",
    "chess piece on white floor",
]

def generate_rgba_images(output_folder="./test_images/rgba"):
    os.makedirs(output_folder, exist_ok=True)
    base_url = "https://image.pollinations.ai/prompt/"

    for i, prompt in enumerate(rgba_prompts):
        url = f"{base_url}{prompt}?height=256&width=256&nologo=true"
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Convert to RGB and resize
            original_img = Image.open(BytesIO(response.content)).convert('RGB').resize((128, 128))

            # Remove background
            img_bytes = BytesIO()
            original_img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            output = remove(img_bytes)  # returns RGBA
            final_img = Image.open(BytesIO(output))

            # Save as PNG (preserves alpha)
            file_path = os.path.join(output_folder, f"rgba_{i+1}.png")
            final_img.save(file_path, format='PNG')
            print(f"✅ Saved: {file_path}")
        except Exception as e:
            print(f"❌ Failed to process image {i+1}: {e}")

generate_rgba_images()
