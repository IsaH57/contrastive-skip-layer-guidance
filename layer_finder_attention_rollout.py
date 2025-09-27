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
    
    num_layers = 0
    for _,_ in enumerate(pipe.transformer.transformer_blocks):
        num_layers += 1

    q_outputs = []
    k_outputs = []

    # Register Hooks
    def register_hooks(transformer):
        def hook_q(module, input, output):
            q_outputs.append(output.detach().cpu())
        def hook_k(module, input, output):
            k_outputs.append(output.detach().cpu())
    
        for i, block in enumerate(transformer.transformer_blocks):
            block.attn.to_q.register_forward_hook(hook_q)
            block.attn.to_k.register_forward_hook(hook_k)

            print(f'Successfully registered QK hooks for block {i}.')
    register_hooks(pipe.transformer)

    all_metrics = {
        "cosine": torch.zeros(len(dataset), args.num_seeds, num_layers),
        "l2": torch.zeros(len(dataset), args.num_seeds, num_layers),
        "abs_activation": torch.zeros(len(dataset), args.num_seeds, num_layers),
    }

    num_steps = 28

    with torch.no_grad():
        for di, pair in enumerate(dataset):
            for seed_i in range(args.num_seeds):
                random_seed = random.randint(0, 2**32 - 1)
                seed_generator = torch.Generator("cpu").manual_seed(random_seed)

                # Forward with target
                _ = pipe(
                    pair["positive"],
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=seed_generator
                )

                q_with = []
                for step in range(num_steps):
                    step_q = q_outputs[step * num_layers : (step + 1) * num_layers]
                    print(f"step_q shape: {step_q[0].shape}")
                    q_with.append(torch.stack(step_q))
                q_with = torch.stack(q_with) 
                q_outputs.clear()

                k_with = []
                for step in range(num_steps):                  
                    step_k = k_outputs[step * num_layers : (step + 1) * num_layers]
                    print(f"step_k shape: {step_k[0].shape}")
                    k_with.append(torch.stack(step_k))
                k_with = torch.stack(k_with)
                k_outputs.clear()

                # Forward without target
                _ = pipe(
                    pair["negative"],
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=seed_generator
                )
                q_without = []
                for step in range(num_steps):
                    step_q = q_outputs[step * num_layers : (step + 1) * num_layers]
                    q_without.append(torch.stack(step_q))
                q_without = torch.stack(q_without)
                q_outputs.clear()

                k_without = []
                for step in range(num_steps):
                    step_k = k_outputs[step * num_layers : (step + 1) * num_layers]
                    k_without.append(torch.stack(step_k))
                k_without = torch.stack(k_without)
                k_outputs.clear()
                
                # Compute per-layer metrics
                for layer in range(num_layers):
                    Qw = q_with[:, layer]
                    Kw = k_with[:, layer]
                    Qn = q_without[:, layer]
                    Kn = k_without[:, layer]

                    QK_with = torch.einsum('sbid,sbjd->sbij', Qw, Kw)
                    print(f"QK shape: {QK_with.shape}")

                    QK_without = torch.einsum('sbid,sbjd->sbij', Qn, Kn)
                    QKdiff = QK_with - QK_without

                    l2_val = torch.norm(QKdiff.view(-1), p=2).item()
                    abs_act = torch.norm(QK_with.view(-1), p=2).item()
                    cos_val = F.cosine_similarity(
                        QK_with.view(-1), 
                        QK_without.view(-1), 
                        dim=0, 
                        eps=1e-8
                    ).item()

                    all_metrics["l2"][di, seed_i, layer] = l2_val
                    all_metrics["abs_activation"][di, seed_i, layer] = abs_act
                    all_metrics["cosine"][di, seed_i, layer] = cos_val

    # Aggregate over dataset and seeds
    avg_metrics_per_layer = {
        metric: all_metrics[metric].mean(dim=(0, 1)) 
        for metric in all_metrics
    }

    # Save full tensor for potential further analysis
    torch.save(all_metrics, f"layer_metrics_{args.model}_{args.target}.pt")

    # Print results
    print("\n📊 Average per-Layer Metrics:")
    for layer in range(num_layers):
        print(
            f"Layer {layer:02d} | "
            f"L2 Dist: {avg_metrics_per_layer['l2'][layer]:.4f} | "
            f"Abs Act: {avg_metrics_per_layer['abs_activation'][layer]:.4f} | "
            f"Cosine: {avg_metrics_per_layer['cosine'][layer]:.4f}"
        )
    
    best_l2_layer = torch.argmax(avg_metrics_per_layer['l2']).item()
    best_abs_act_layer = torch.argmax(avg_metrics_per_layer['abs_activation']).item()
    best_cosine_layer = torch.argmax(avg_metrics_per_layer['cosine']).item()

    print("\n🏆 Best Layers by Metric:")
    print(f"Highest L2 Distance: Layer {best_l2_layer:02d} with {avg_metrics_per_layer['l2'][best_l2_layer]:.4f}")
    print(f"Highest Absolute Activation: Layer {best_abs_act_layer:02d} with {avg_metrics_per_layer['abs_activation'][best_abs_act_layer]:.4f}")
    print(f"Highest Cosine Similarity: Layer {best_cosine_layer:02d} with {avg_metrics_per_layer['cosine'][best_cosine_layer]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', etc.")
    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")

    args = parser.parse_args()
    main(args)