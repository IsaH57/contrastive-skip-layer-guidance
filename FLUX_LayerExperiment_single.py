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
dataset_path = os.path.join(os.getcwd(), "prompt_datasets/prompts_text.json")
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
    for i, prompt in enumerate(dataset[:3]):
        num_steps = random.randint(1,3)

        _ = pipe(
            prompt,
            num_inference_steps=num_steps,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        )

        layer_outputs_with_text = layer_outputs.copy()[-19:]
        layer_outputs.clear()
        
        if len(layer_outputs_with_text) != 19:
            print('ALARM')
        
        # Compare layer outputs using cosine similarity
        similarities = []
        for out_current, out_next in zip(layer_outputs_with_text[:-1], layer_outputs_with_text[1:]):
            # Flatten and compute cosine similarity
            cos_sim = F.cosine_similarity(
                out_current.flatten(start_dim=1),
                out_next.flatten(start_dim=1),
                dim=1
            ).mean().item()
            similarities.append(cos_sim)
        pass_wise_similartites.append(similarities)

        # Find layer with lowest cosine similarity
        min_sim = min(similarities)
        min_index = similarities.index(min_sim)
        # Print results
        print(f"🔍 \n For prompt with text: {prompt}")
        print(f"Layer {min_index} has lowest Cosine Similarity of {min_sim:.4f}")


similarity_tensor = torch.tensor(pass_wise_similartites)  # shape: (num_passes, num_layers)
torch.save(similarity_tensor, "similarity_tensor_full_dataset.pt")

# Average across passes first
mean_similarities = similarity_tensor.mean(dim=0)

# Apply softmax to normalized average
softmaxed_avg = F.softmax(mean_similarities, dim=0)

# Print result
print("\n📊 Global Average (then Softmax) Cosine Similarity per Layer:")
for i, score in enumerate(softmaxed_avg.tolist()):
    print(f"Layer {i:02d}: {score:.4f}")
