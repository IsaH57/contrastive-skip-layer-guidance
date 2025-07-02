""" This script uses the build-in layer skipping function of a Stable Diffusion 3 model to generate images with varying guidance scales."""

import gc
import os
import torch
from diffusers import StableDiffusion3Pipeline as SD3Pipeline

SKIPPED_LAYERS = [9, 12]  # Specify which layers to skip
GUIDANCE_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]  # Guidance scale values to test
prompt= "An infographic poster listing '20 Ways to Reduce Your Carbon Footprint', each tip in its own box with text and a small icon, arranged in 4 columns"

# Create results directory
results_dir = os.path.join(os.getcwd(), f"sd3_slg_buildin_carbon_footprint")
os.makedirs(results_dir, exist_ok=True)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# Custom pipeline with layer skipping
pipe = SD3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe.to("cuda")

for guidance_scale in GUIDANCE_SCALES:
    # Generate image with layer skipping
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        skip_guidance_layers=SKIPPED_LAYERS,
        num_inference_steps=20
    ).images[0]

    image.save(f"{results_dir}/carbon_footprint_slg_{guidance_scale}.png")
    flush()

    # Generate without skipping layers for comparison
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=20
    ).images[0]

    image.save(f"{results_dir}/carbon_footprint_{guidance_scale}.png")
    flush()