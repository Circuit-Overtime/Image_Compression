import requests
from PIL import Image
from io import BytesIO
import os

prompts = [
    " portrait sketch of a wise old man",
    " cityscape with deep shadows",
    " forest scene with fog",
    " astronaut floating in space",
    " cat on a windowsill",
    " mountain range with clouds",
    " sketch of a dancer mid-move",
    "vintage  photo of a train station",
    " underwater scene with fish silhouettes",
    " profile of a woman with flowing hair",
    " chessboard with dramatic lighting",
    " ruins of an ancient castle",
    " cyberpunk alleyway",
    " violin on a wooden table",
    " sketch of a dragon curled up",
    " tree in the middle of a field",
    " haunted house on a hill",
    " owl perched on a branch",
    " bike leaning on a wall",
    " painting of a lighthouse by the sea",
]

def generate_rgb_images(output_folder="./test_images/rgb"):
    os.makedirs(output_folder, exist_ok=True)
    base_url = "https://image.pollinations.ai/prompt/"

    for i, prompt in enumerate(prompts):
        url = f"{base_url}{prompt}?height=256&width=256&nologo=true"
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Convert to RGB and resize
            img = Image.open(BytesIO(response.content)).convert('RGB').resize((128, 128))

            file_path = os.path.join(output_folder, f"rgb_{i+1}.jpg")
            img.save(file_path, format='JPEG')
            print(f"✅ Saved: {file_path}")
        except Exception as e:
            print(f"❌ Failed to fetch image {i+1}: {e}")

generate_rgb_images()
