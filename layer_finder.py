"""
FLUX[dev] - Skip Layer Guidance Exploration

This script generates images for pairs of prompts that differ only 
in the target concept being present in the image. We then compare intermediate 
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
import argparse
import random

def main(args):

    # Load dataset of prompt pairs 
    dataset = json.load(open(args.dataset_path, "r"))

    # Load model
    if args.model == 'flux':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    elif args.model == 'sd3':
        raise NotImplementedError()
    else:
        raise NameError('Invalid model name.')

    # Store intermediate outputs
    layer_outputs = []

    # Register Hooks
    def register_hooks(transformer):
        def hook_fn(module, input, output):
            layer_outputs.append(output.detach().cpu())
        # Iterate over all Transformer blocks
        for i, block in enumerate(transformer.transformer_blocks):
            block.norm2.register_forward_hook(hook_fn)
            print(f'Successfully registered hook for block {i}.')
    register_hooks(pipe.transformer)

    pass_wise_similartites = []
    # Example forward pass 
    with torch.no_grad():
        for i, pair in enumerate(dataset):
            prompt_positive = pair["positive"]
            prompt_negative = pair["negative"]
            for seed in range(args.num_seeds):
                num_steps = random.randint(1,28)
                random_seed = random.randint(0, 2**32 - 1)
                seed_generator = torch.Generator("cpu").manual_seed(random_seed)
                _ = pipe(
                    prompt_positive,
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=seed_generator
                )

                layer_outputs_with_text = layer_outputs.copy()[-19:] #TODO: variable length
                layer_outputs.clear()

                _ = pipe(
                    prompt_negative,
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=seed_generator
                )
                layer_outputs_no_text = layer_outputs.copy()[-19:] #TODO: variable length
                layer_outputs.clear()
                
                print(layer_outputs_with_text)
                print(layer_outputs_no_text)

                if len(layer_outputs_no_text) != 19 or len(layer_outputs_with_text) != 19: #TODO: variable length
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
    torch.save(similarity_tensor, f"similarity_tensor_{args.model}_{args.target}.pt")

    # Average across passes first
    mean_similarities = similarity_tensor.mean(dim=0)

    # Apply softmax to normalized average
    softmaxed_avg = F.softmax(mean_similarities, dim=0)

    # Print result
    print("\n📊 Global Average (then Softmax) Cosine Similarity per Layer:")
    for i, score in enumerate(softmaxed_avg.tolist()):
        print(f"Layer {i:02d}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', etc.")
    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")

    args = parser.parse_args()
    main(args)