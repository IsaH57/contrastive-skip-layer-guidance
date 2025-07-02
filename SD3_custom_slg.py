""" This script uses an own skipping function for a Stable Diffusion 3 model to generate images with varying guidance scales.
Note: It does not work as intended and only produces noise!
"""
from datetime import datetime
import os

import torch
from SD3_custom_pipeline import StableDiffusion3Pipeline as CustomSD3Pipeline

SKIPPED_LAYERS = [9, 12]  # Specify which layers to skip
GUIDANCE_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]  # Guidance scale values to test

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), f"sd3_slg_custom_container")
os.makedirs(results_dir, exist_ok=True)

# Custom pipeline with layer skipping
pipe = CustomSD3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

for guidance_scale in GUIDANCE_SCALES:
    print(f"Generating image with guidance scale: {guidance_scale}")
    image = pipe(
        prompt="a shipping container on a dock near the water with the word 'EXPORT' painted on it",
        guidance_scale=guidance_scale,
        skipped_layers=SKIPPED_LAYERS,
        num_inference_steps=20,

    ).images[0]

    image.save(f"{results_dir}/container{guidance_scale}.png")
