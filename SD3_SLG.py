"""
Stable Diffusion 3 - Skip Layer Guidance Exploration
See https://huggingface.co/stabilityai/stable-diffusion-3-medium

This script generates images for pairs of prompts that differ only 
in text being present in the image. We then compare intermediate 
outputs of the model to extract information about which layers 
correspond most to text being present in the image.
"""

import gc
import json 
import inspect
import os
from pathlib import Path
import torch
from diffusers import StableDiffusion3Pipeline

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), "prompts_text_notext.json")
dataset = json.load(open(dataset_path, "r"))

# Load the pipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

layer_outputs = []
def register_hooks(model):
    def hook_fn(module, input, output):
        layer_outputs.append(output.detach().cpu())

    # Find Transformer blocks inside UNet
    for i, block in enumerate(model.unet.transformer_blocks):
        block.register_forward_hook(hook_fn)

for i, pair in enumerate(dataset):
    prompt_text = pair["wit_text"]
    prompt_notext = pair["no_text"]