"""This script generates images for pairs of prompts that differ only
in text being present in the image. We then compare intermediate
outputs of the model to extract information about which layers
correspond most to text being present in the image.
"""

import os

from SD3_Pipeline_Tests.SD3_custom_pipeline import StableDiffusion3Pipeline
import torch
from datetime import datetime

# Parameters
skip_layer_indices = [9, 12]  # Specify which layers to skip here
gs = [0.0, 0.5, 1.0, 3.0, 5.0, 7.5, 10.0]  # Guidance scale values to test

prompt = "An infographic poster listing '20 Ways to Reduce Your Carbon Footprint', each tip in its own box with text and a small icon, arranged in 4 columns"

# Load pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16)
pipe.to("cuda")

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), f"sd3_skip_results_carbon_footprint")
os.makedirs(results_dir, exist_ok=True)

# Process prompts
for s in [0.0, 0.5, 1.0, 3.0, 5.0, 7.5, 10.0]:
# Modified generation with skip layers
    pipe.skip_layers = skip_layer_indices
    image_modified = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=s).images[0]
    image_modified_path = os.path.join(results_dir, f"cfgsacle_{s}.png")
    image_modified.save(image_modified_path)

    print(f"Saved:, {image_modified_path}")
