prompts = [
    "grayscale portrait sketch of a wise old man",
"black and white cityscape with deep shadows",
"grayscale forest scene with fog",
"grayscale astronaut floating in space",
"black and white cat on a windowsill",
"grayscale mountain range with clouds",
"grayscale sketch of a dancer mid-move",
"vintage grayscale photo of a train station",
"grayscale underwater scene with fish silhouettes",
"grayscale profile of a woman with flowing hair",
"black and white chessboard with dramatic lighting",
"grayscale ruins of an ancient castle",
"grayscale cyberpunk alleyway",
"grayscale violin on a wooden table",
"grayscale sketch of a dragon curled up",
"black and white tree in the middle of a field",
"grayscale haunted house on a hill",
"grayscale owl perched on a branch",
"black and white bike leaning on a wall",
"grayscale painting of a lighthouse by the sea",
]

import requests
from PIL import Image
from io import BytesIO
import os

def generate_grayscale_images(output_folder="./test_images/grayscale"):
    os.makedirs(output_folder, exist_ok=True)
    base_url = "https://image.pollinations.ai/prompt/"

    for i in range(len(prompts)):
        prompt = f"{prompts[i]}"  # make prompts a bit unique
        url = f"{base_url}{prompt}?height=256&width=256&nologo=true"
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Convert to grayscale and resize
            img = Image.open(BytesIO(response.content)).convert('L')
            img = img.resize((128, 128))    
            

            file_path = os.path.join(output_folder, f"grayscale_{i+1}.jpg")
            img.save(file_path, format='JPEG')
            print(f"✅ Saved: {file_path}")
        except Exception as e:
            print(f"❌ Failed to fetch image {i+1}: {e}")

generate_grayscale_images()
