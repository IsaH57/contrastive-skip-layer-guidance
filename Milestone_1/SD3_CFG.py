"""
Stable Diffusion 3 - CFG Guidance Scale Exploration
See https://huggingface.co/stabilityai/stable-diffusion-3-medium

This script generates images from the same prompt using different 
Classifier-Free Guidance (CFG) scales to show the effect 
of differnet guidance weights on the image output.
"""

import gc
import inspect
import os
from pathlib import Path
import torch
from diffusers import StableDiffusion3Pipeline

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create output dir
output_dir = os.path.join(os.getcwd(), f"img_{file_path}")
os.makedirs(output_dir, exist_ok=True)

# Def prompt
PROMPT = "an ancient temple in the jungle, covered in moss, with golden statues and shafts of light piercing through the trees"

# List of guidance scales to test
GUIDANCE_SCALES = [0., 0.5, 1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]

# Load the pipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for gs in GUIDANCE_SCALES:
    print(f"Generating image with CFG scale: {gs}")

    # Run the pipeline with specified CFG
    with torch.no_grad():
        image = pipe(
            PROMPT,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=gs,
        ).images[0]

    # Save image
    image_path = os.path.join(output_dir, f"cfgscale_{gs:.1f}.png")
    image.save(image_path)

# Cleanup 
def flush():
    gc.collect()
    torch.cuda.empty_cache()
flush()