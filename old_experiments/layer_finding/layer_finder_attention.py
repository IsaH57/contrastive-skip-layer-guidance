"""
FLUX[dev] - Skip Layer Guidance Exploration

This script generates images for pairs of prompts that differ only 
in the target concept being present in the image. We then compare intermediate 
outputs of the model to extract information about which layers 
correspond most to text being present in the image.
"""

import gc
import json 
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
import argparse
import random

def safe_cosine(a, b, eps=1e-8, nan_to=0.0):
    vec1 = a.reshape(-1)
    vec2 = b.reshape(-1)
    if torch.norm(vec1).item() == 0.0 and torch.norm(vec2).item() == 0.0:
        return 1.0
    if torch.norm(vec1).item() == 0.0 or torch.norm(vec2).item() == 0.0:
        return 0.0
    cos = F.cosine_similarity(vec1, vec2, dim=0, eps=eps)
    if torch.isnan(cos):
        raise ValueError('Cosine Similarity is None.')
    return cos.item()

def main(args):

    # Load dataset of prompt pairs 
    dataset = json.load(open(args.dataset_path, "r"))

    # Load model
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    num_layers = len(pipe.transformer.transformer_blocks)
    num_steps = 28

    # Register hooks
    qs_img = []
    ks_img = []
    qs_txt = []
    ks_txt = []

    def register_hooks(transformer):
        # Hook normalized queries and keys for image tokens
        def hook_img_q(module, input, output):
            qs_img.append(output.detach())  
        def hook_img_k(module, input, output):
            ks_img.append(output.detach())  
        # Hook normalized queries and keys for text tokens
        def hook_txt_q(module, input, output):
            qs_txt.append(output.detach())  
        def hook_txt_k(module, input, output):
            ks_txt.append(output.detach())  
        for i, block in enumerate(transformer.transformer_blocks):
            block.attn.norm_q.register_forward_hook(hook_img_q)
            block.attn.norm_k.register_forward_hook(hook_img_k)
            block.attn.norm_added_q.register_forward_hook(hook_txt_q)
            block.attn.norm_added_k.register_forward_hook(hook_txt_k)
            print(f'Successfully registered QK hooks for block {i}.')

    register_hooks(pipe.transformer)

    # Set up metrics
    all_metrics = {
        "cos_t2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "cos_t2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "cos_i2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "cos_i2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "abs_t2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "abs_t2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "abs_i2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "abs_i2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "l2_t2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "l2_t2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "l2_i2t": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        "l2_i2i": torch.zeros(len(dataset), args.num_seeds, num_layers, num_steps),
        }

    def cleanup(): 
        qs_img.clear()
        ks_img.clear()
        qs_txt.clear()
        ks_txt.clear()
        torch.cuda.empty_cache()
        gc.collect()

    with torch.no_grad():
        for di, pair in enumerate(dataset):
            for seed_i in range(args.num_seeds):
                random_seed = random.randint(0, 2**32 - 1)

                # Forward with positive prompt (with target)
                _ = pipe(
                    pair["positive"],
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(random_seed)
                )     

                q_img_p = [q for q in qs_img]
                k_img_p = [k for k in ks_img]
                q_txt_p = [q for q in qs_txt]
                k_txt_p = [k for k in ks_txt]
                cleanup()


                # Forward with negative prompt (without target)
                _ = pipe(
                    pair["negative"],
                    num_inference_steps=num_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(random_seed)
                )

                q_img_n = [q for q in qs_img]
                k_img_n = [k for k in ks_img]
                q_txt_n = [q for q in qs_txt]
                k_txt_n = [k for k in ks_txt]
                cleanup()

                for layer in range(num_layers):
                    for step in range(num_steps):
                        Q_img_p = q_img_p[step * num_layers + layer]
                        K_img_p = k_img_p[step * num_layers + layer]
                        Q_txt_p = q_txt_p[step * num_layers + layer]
                        K_txt_p = k_txt_p[step * num_layers + layer]

                        Q_img_n = q_img_n[step * num_layers + layer]
                        K_img_n = k_img_n[step * num_layers + layer]
                        Q_txt_n = q_txt_n[step * num_layers + layer]
                        K_txt_n = k_txt_n[step * num_layers + layer]

                        Q_merged_p = torch.cat([Q_txt_p, Q_img_p], dim=1)  
                        K_merged_p = torch.cat([K_txt_p, K_img_p], dim=1)         
                        Q_merged_n = torch.cat([Q_txt_n, Q_img_n], dim=1)  
                        K_merged_n = torch.cat([K_txt_n, K_img_n], dim=1)   

                        del Q_img_p, K_img_p, Q_txt_p, K_txt_p, Q_img_n, K_img_n, Q_txt_n, K_txt_n

                        QK_p_merged = torch.einsum('bihd,bjhd->bhij', Q_merged_p, K_merged_p) / (Q_merged_p.shape[-1] ** 0.5)
                        QK_n_merged = torch.einsum('bihd,bjhd->bhij', Q_merged_n, K_merged_n) / (Q_merged_n.shape[-1] ** 0.5)
                        QK_p = torch.softmax(QK_p_merged, dim=-1)
                        QK_n = torch.softmax(QK_n_merged, dim=-1)

                        QK_diff = QK_p - QK_n                      

                        # Calculate Metrics
                        # Text-2-Text
                        all_metrics["l2_t2t"][di, seed_i, layer, step] = torch.sqrt(torch.sum(QK_diff[:, :, :256, :256] ** 2)).item()
                        all_metrics["abs_t2t"][di, seed_i, layer, step] = (torch.sum(QK_p[:, :, :256, :256]) - torch.sum(QK_n[:, :, :256, :256])).item()    
                        all_metrics["cos_t2t"][di, seed_i, layer, step] = safe_cosine(
                            QK_p[:, :, :256, :256].reshape(-1),
                            QK_n[:, :, :256, :256].reshape(-1),
                        )

                        # Text-2-Image
                        all_metrics["l2_t2i"][di, seed_i, layer, step] = torch.sqrt(torch.sum(QK_diff[:, :, :256, 256:] ** 2)).item()
                        all_metrics["abs_t2i"][di, seed_i, layer, step] = (torch.sum(QK_p[:, :, :256, 256:]) - torch.sum(QK_n[:, :, :256, 256:])).item()    
                        all_metrics["cos_t2i"][di, seed_i, layer, step] = safe_cosine(
                            QK_p[:, :, :256, 256:].reshape(-1),
                            QK_n[:, :, :256, 256:].reshape(-1),
                        )

                        # Image-2-Text
                        all_metrics["l2_i2t"][di, seed_i, layer, step] = torch.sqrt(torch.sum(QK_diff[:, :, 256:, :256] ** 2)).item()
                        all_metrics["abs_i2t"][di, seed_i, layer, step] = (torch.sum(QK_p[:, :, 256:, :256]) - torch.sum(QK_n[:, :, 256:, :256])).item()    
                        all_metrics["cos_i2t"][di, seed_i, layer, step] = safe_cosine(
                            QK_p[:, :, 256:, :256].reshape(-1),
                            QK_n[:, :, 256:, :256].reshape(-1),
                        )

                        # Image-2-Image
                        all_metrics["l2_i2i"][di, seed_i, layer, step] = torch.sqrt(torch.sum(QK_diff[:, :, 256:, 256:] ** 2)).item()
                        all_metrics["abs_i2i"][di, seed_i, layer, step] = (torch.sum(QK_p[:, :, 256:, 256:]) - torch.sum(QK_n[:, :, 256:, 256:])).item()    
                        all_metrics["cos_i2i"][di, seed_i, layer, step] = safe_cosine(
                            QK_p[:, :, 256:, 256:].reshape(-1),
                            QK_n[:, :, 256:, 256:].reshape(-1),
                        )

                        del QK_p_merged, QK_n_merged, QK_p, QK_n, QK_diff
                        torch.cuda.empty_cache()
                        gc.collect()

                # Clean up GPU tensors after processing this seed
                del q_img_p, k_img_p, q_txt_p, k_txt_p
                del q_img_n, k_img_n, q_txt_n, k_txt_n
                torch.cuda.empty_cache()
                gc.collect()

    # Save full tensor 
    torch.save(all_metrics, f"layer_metrics_{args.model}_{args.target}.pt")

    # Aggregate 
    avg_metrics_per_layer = {
        metric: all_metrics[metric].mean(dim=(0, 1, 3)) 
        for metric in all_metrics
    }

    # Print results
    print("\n📊 Average per-Layer Metrics:")
    for metric in avg_metrics_per_layer.keys():
        print(f"\n=== {metric} ===")
        for layer_idx, _ in enumerate(avg_metrics_per_layer[metric]):
            print(f"Layer_{layer_idx}: {avg_metrics_per_layer[metric][layer_idx]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', etc.")
    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")

    args = parser.parse_args()
    main(args)