import os
import json
import torch
import gc
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import re
import random
import torch.nn.functional as F
import argparse

from FLUX_custom_pipeline import FluxPipeline
from FLUX_custom_transformer import FluxTransformer2DModel

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    dataset = json.load(open(args.dataset_path, "r"))

    # Load custom transformer
    custom_transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.float16
    )

    # Load pipeline with custom transformer
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=custom_transformer,
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.activation_patching = True

    layer_diffs=[]
    cosine_sims=[]
    layer_diffs_normalized=[]
    cosine_sims_normalized=[]

    for prompt_pair in dataset[:args.max_prompts]:
        for seed in range(args.num_seeds):
            print(f"Generating image for negative prompt: {prompt_pair['negative']} ")
            print(f"Injecting positive prompt: {prompt_pair['positive']} ")

            pipe.skipped_latents = []
            pipe.unskipped_latents = []
            pipe.patch_prompt = prompt_pair['positive']

            _ = pipe(
                prompt=prompt_pair['negative'],
                num_inference_steps=args.num_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1))
            ).images[0]

            layer_diff_for_prompt = []
            cosine_sims_for_prompt = []
            layer_diff_for_prompt_normalized = []
            cosine_sims_for_prompt_normalized = []

            for timestep in range(args.num_steps): 
                layer_diffs_per_timestep = []
                cosine_sims_per_timestep = []
                layer_diffs_per_timestep_normalized = []
                cosine_sims_per_timestep_normalized = []

                timestep_prediction = pipe.unskipped_latents[timestep]
                timestep_baseline = pipe.skipped_latents[timestep][19]
                diff_baseline = (timestep_prediction - timestep_baseline)
                squared_norm_baseline = torch.sum(diff_baseline ** 2, dim=[0,1,2])
                sim_baseline = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][19].flatten(), dim=0)

                for layer in range(19):
                    diff = (timestep_prediction - pipe.skipped_latents[timestep][layer])
                    squared_norm = torch.sum((diff ** 2), dim=[0,1,2])
                    cosine_sim = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][layer].flatten(), dim=0)

                    layer_diffs_per_timestep.append(squared_norm.item())
                    cosine_sims_per_timestep.append(cosine_sim.item())

                    layer_diffs_per_timestep_normalized.append((squared_norm / squared_norm_baseline).item())
                    cosine_sims_per_timestep_normalized.append((cosine_sim / sim_baseline).item())

                layer_diff_for_prompt.append(layer_diffs_per_timestep)
                cosine_sims_for_prompt.append(cosine_sims_per_timestep)

                layer_diff_for_prompt_normalized.append(layer_diffs_per_timestep_normalized)
                cosine_sims_for_prompt_normalized.append(cosine_sims_per_timestep_normalized)

            layer_diffs.append(layer_diff_for_prompt)
            cosine_sims.append(cosine_sims_for_prompt) 

            layer_diffs_normalized.append(layer_diff_for_prompt_normalized)
            cosine_sims_normalized.append(cosine_sims_for_prompt_normalized) 

        # Clean up
        flush()

    metrics = {
        'layer_diffs': layer_diffs,
        'cosine_sims': cosine_sims,
        'layer_diffs_normalized': layer_diffs_normalized,
        'cosine_sims_normalized': cosine_sims_normalized
    }

    # Save results
    output_path = f'AP_metrics_pos_inj_{args.target}.json'
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


    layer_diffs=[]
    cosine_sims=[]
    layer_diffs_normalized=[]
    cosine_sims_normalized=[]

    for prompt_pair in dataset[:args.max_prompts]:
        for seed in range(args.num_seeds):
            print(f"Generating image for negative prompt: {prompt_pair['positive']} ")
            print(f"Injecting positive prompt: {prompt_pair['negative']} ")

            pipe.skipped_latents = []
            pipe.unskipped_latents = []
            pipe.patch_prompt = prompt_pair['negative']

            _ = pipe(
                prompt=prompt_pair['positive'],
                num_inference_steps=args.num_steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1))
            ).images[0]

            layer_diff_for_prompt = []
            cosine_sims_for_prompt = []
            layer_diff_for_prompt_normalized = []
            cosine_sims_for_prompt_normalized = []

            for timestep in range(args.num_steps): 
                layer_diffs_per_timestep = []
                cosine_sims_per_timestep = []
                layer_diffs_per_timestep_normalized = []
                cosine_sims_per_timestep_normalized = []

                timestep_prediction = pipe.unskipped_latents[timestep]
                timestep_baseline = pipe.skipped_latents[timestep][19]
                diff_baseline = (timestep_prediction - timestep_baseline)
                squared_norm_baseline = torch.sum(diff_baseline ** 2, dim=[0,1,2])
                sim_baseline = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][19].flatten(), dim=0)

                for layer in range(19):
                    diff = (timestep_prediction - pipe.skipped_latents[timestep][layer])
                    squared_norm = torch.sum((diff ** 2), dim=[0,1,2])
                    cosine_sim = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][layer].flatten(), dim=0)

                    layer_diffs_per_timestep.append(squared_norm.item())
                    cosine_sims_per_timestep.append(cosine_sim.item())

                    layer_diffs_per_timestep_normalized.append((squared_norm / squared_norm_baseline).item())
                    cosine_sims_per_timestep_normalized.append((cosine_sim / sim_baseline).item())

                layer_diff_for_prompt.append(layer_diffs_per_timestep)
                cosine_sims_for_prompt.append(cosine_sims_per_timestep)

                layer_diff_for_prompt_normalized.append(layer_diffs_per_timestep_normalized)
                cosine_sims_for_prompt_normalized.append(cosine_sims_per_timestep_normalized)

            layer_diffs.append(layer_diff_for_prompt)
            cosine_sims.append(cosine_sims_for_prompt) 

            layer_diffs_normalized.append(layer_diff_for_prompt_normalized)
            cosine_sims_normalized.append(cosine_sims_for_prompt_normalized) 

        # Clean up
        flush()

    metrics = {
        'layer_diffs': layer_diffs,
        'cosine_sims': cosine_sims,
        'layer_diffs_normalized': layer_diffs_normalized,
        'cosine_sims_normalized': cosine_sims_normalized
    }

    # Save results
    output_path = f'AP_metrics_neg_inj_{args.target}.json'
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', etc.")
    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--num_steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of seeds per prompt.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--max_prompts", type=int, default=100)


    args = parser.parse_args()
    main(args)