"""
FLUX[dev] - Skip Layer Guidance Exploration
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
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from diffusers import FluxPipeline
import random

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), "prompt_datasets/hands/hands.json")
dataset = json.load(open(dataset_path, "r"))

# Load the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()  
flux_transformer = pipe.transformer

# Store intermediate outputs
layer_outputs = []

# Register Hooks
def register_hooks(transformer):
    def hook_fn(module, input, output):
        layer_outputs.append(output.detach().cpu())
    # Iterate over all Transformer blocks
    for i, block in enumerate(transformer.transformer_blocks):
        if hasattr(block.attn, "to_out") and isinstance(block.attn.to_out[0], torch.nn.Linear):
            block.attn.to_out[0].register_forward_hook(hook_fn)
register_hooks(flux_transformer)

pass_wise_similartites = []
# Example forward pass 
with torch.no_grad():
    for i, pair in enumerate(dataset):
        prompt_positive = pair["positive"]
        prompt_negative = pair["negative"]
        for seed in range(1):
            num_steps = random.randint(1,28)
            random_seed_a = random.randint(0, 2**32 - 1)
            random_seed_b = random.randint(0, 2**32 - 1)

            pos = pipe(
                '',
                num_inference_steps=num_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random_seed_a)
            )

            layer_outputs_with_text = layer_outputs.copy()[-19:]
            layer_outputs.clear()

            neg = pipe(
                '',
                num_inference_steps=num_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random_seed_b)
            )
            layer_outputs_no_text = layer_outputs.copy()[-19:]
            layer_outputs.clear()
            
            if len(layer_outputs_no_text) != 19 or len(layer_outputs_with_text) != 19:
                print('ALARM')
            
            # Compare layer outputs using cosine similarity
            similarities = []
            for out_text, out_no_text in zip(layer_outputs_with_text, layer_outputs_no_text):
                # Flatten and compute cosine similarity
                cos_sim = F.cosine_similarity(
                    out_text.flatten(start_dim=1),
                    out_no_text.flatten(start_dim=1),
                    dim=1
                ).mean().item()
                similarities.append(cos_sim)
        pass_wise_similartites.append(similarities)


similarity_tensor = torch.tensor(pass_wise_similartites)  # shape: (num_passes, num_layers)
torch.save(similarity_tensor, "similarity_tensor_FLUX_base.pt")

# Average across passes first
mean_similarities = similarity_tensor.mean(dim=0)

# Apply softmax to normalized average
softmaxed_avg = F.softmax(mean_similarities, dim=0)

# Print result
print("\n📊 Global Average (then Softmax) Cosine Similarity per Layer:")
for i, score in enumerate(softmaxed_avg.tolist()):
    print(f"Layer {i:02d}: {score:.4f}")
