""" This script generates images for pairs of prompts that differ only in text being present in the image.
We then compare intermediate outputs of the model to extract information about which layers correspond most to text being present in the image.
"""

import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import random

from diffusers import StableDiffusion3Pipeline

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Load dataset of prompt pairs
dataset_path = os.path.join(os.getcwd(), "prompt_datasets/prompts_text_notext.json")
dataset = json.load(open(dataset_path, "r"))

# Load the pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
sd3_transformer = pipe.transformer

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


register_hooks(sd3_transformer)

pass_wise_similartites = []
# Example forward pass
with torch.no_grad():
    for i, pair in enumerate(dataset):
        prompt_text = pair["with_text"]
        prompt_notext = pair["no_text"]  #
        num_steps = random.randint(1, 20)

        _ = pipe(
            prompt_text,
            num_inference_steps=num_steps,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        )

        layer_outputs_with_text = layer_outputs.copy()[-19:]
        layer_outputs.clear()

        b = pipe(
            prompt_notext,
            num_inference_steps=num_steps,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
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

        # Find layer with lowest cosine similarity
        min_sim = min(similarities)
        min_index = similarities.index(min_sim)
        # Print results
        print(f"🔍 \n For prompt with text: {prompt_text} \n and without text: {prompt_notext}")
        print(f"Layer {min_index} has lowest Cosine Similarity of {min_sim:.4f}")

similarity_tensor = torch.tensor(pass_wise_similartites)  # shape: (num_passes, num_layers)
torch.save(similarity_tensor, "SD3_similarity_tensor_full_dataset.pt")

# Average across passes first
mean_similarities = similarity_tensor.mean(dim=0)

# Apply softmax to normalized average
softmaxed_avg = F.softmax(mean_similarities, dim=0)

# Print result
print("\nGlobal Average (then Softmax) Cosine Similarity per Layer:")
for i, score in enumerate(softmaxed_avg.tolist()):
    print(f"Layer {i:02d}: {score:.4f}")


def plot_relative_similarity_change(similarity_tensor: torch.Tensor):
    """
    Plots the relative change in cosine similarity for each layer across all prompt pairs.

    Args:
        similarity_tensor (torch.Tensor): A tensor of shape (num_pairs, num_layers) containing cosine similarity values.
    """

    similarity_array = similarity_tensor

    # Calculate relative changes
    relative_changes = np.zeros_like(similarity_array)
    for i in range(similarity_array.shape[0]):
        for j in range(1, similarity_array.shape[1]):
            prev_val = similarity_array[i, j - 1]
            if prev_val != 0:  # Division durch Null vermeiden
                relative_changes[i, j] = (similarity_array[i, j] - prev_val) / prev_val
            else:
                relative_changes[i, j] = 0

    # Set the first layer's relative change to 0 (as there's no previous layer)
    relative_changes[:, 0] = 0

    plt.figure(figsize=(12, 8))

    # Plot each pair's relative changes in light gray
    for i in range(relative_changes.shape[0]):
        plt.plot(range(1, relative_changes.shape[1]), relative_changes[i, 1:],
                 color='lightgray', linewidth=0.5)

    # Plot the mean relative changes in red
    mean_changes = relative_changes.mean(axis=0)
    plt.plot(range(1, len(mean_changes)), mean_changes[1:],
             color='red', linewidth=2.5, marker='o')

    plt.title("Relative Change in Cosine Similarity per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Relative change in Cosine Similarity from previous layer")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, similarity_array.shape[1]),
               [f"{i}" for i in range(1, similarity_array.shape[1])])

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig("SD3_relative_similarity_change.png", dpi=300)
    plt.show()


plot_relative_similarity_change(similarity_tensor)
