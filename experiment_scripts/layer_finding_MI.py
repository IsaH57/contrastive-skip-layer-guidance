import json
import torch
import gc
import sys
import os
import random
import torch.nn.functional as F
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.FLUX_custom_pipeline import FluxPipeline
from src.SD3_custom_pipeline import StableDiffusion3Pipeline

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):

    print(f'PERFORMING LAYER SEARCH FOR TARGET: {args.target}')
    print(f'USING MODEL: {args.model}')
    print(f'MAX PROMPTS: {args.max_prompts}')

    # Load Dataset
    match args.target: 
        case 'text': 
            dataset = json.load(open('prompt_datasets/text_pairs.json', "r"))
        case 'hands':
            dataset = json.load(open('prompt_datasets/hands_pairs.json', "r"))
        case 'aesthetics':
            dataset = json.load(open('prompt_datasets/aesthetics_pairs.json', "r"))
        case _: 
            if args.dataset_path != '':
                dataset = json.load(open(args.dataset_path, "r"))
            else:
                raise ValueError()
    dataset = random.shuffle(dataset)

    # Load Pipeline
    match args.model: 
        case 'flux':
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        case 'sd3':
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        case 'pixart':
            raise NotImplementedError()
    pipe.to("cuda")
    pipe.layer_search = True

    # --- METRIC LISTS ---
    layer_diffs = []
    cosine_sims = []
    layer_diffs_normalized = []
    cosine_sims_normalized = []
    layer_diffs_subtracted = []
    cosine_sims_subtracted = []

    for idx, prompt_pair in enumerate(dataset[:args.max_prompts]):
        for seed in range(args.num_seeds):
            print(f"Generating image for negative prompt: {prompt_pair['negative']} ")
            print(f"Injecting positive prompt: {prompt_pair['positive']} ")

            pipe.skipped_latents = []
            pipe.unskipped_latents = []
            pipe.patch_prompt = prompt_pair['positive']

            _ = pipe(
                prompt=prompt_pair['negative'],
                num_inference_steps=args.num_steps,
                generator=torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1))
            ).images[0]

            # --- Lists for this specific prompt/seed run ---
            layer_diff_for_prompt = []
            cosine_sims_for_prompt = []
            layer_diff_for_prompt_normalized = []
            cosine_sims_for_prompt_normalized = []
            layer_diff_for_prompt_subtracted = []
            cosine_sims_for_prompt_subtracted = []

            for timestep in range(args.num_steps): 
                # --- Lists for this specific timestep ---
                layer_diffs_per_timestep = []
                cosine_sims_per_timestep = []
                layer_diffs_per_timestep_normalized = []
                cosine_sims_per_timestep_normalized = []
                layer_diffs_per_timestep_subtracted = []
                cosine_sims_per_timestep_subtracted = []

                # --- Baseline Calculations ---
                timestep_prediction = pipe.unskipped_latents[timestep]
                timestep_baseline = pipe.skipped_latents[timestep][-1]
                diff_baseline = (timestep_prediction - timestep_baseline)
                
                # We calculate the norm of the difference vector.
                l2_norm_baseline = torch.norm(diff_baseline.float(), p=2) 
                sim_baseline = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][-1].flatten(), dim=0)

                for layer in range(19):
                    diff = (timestep_prediction - pipe.skipped_latents[timestep][layer])
                    
                    # --- Individual Layer Calculations ---
                    l2_norm = torch.norm(diff.float(), p=2)
                    cosine_sim = F.cosine_similarity(timestep_prediction.flatten(), pipe.skipped_latents[timestep][layer].flatten(), dim=0)

                    # 1. Absolute metrics
                    layer_diffs_per_timestep.append(l2_norm.item())
                    cosine_sims_per_timestep.append(cosine_sim.item())

                    # 2. Ratio-normalized metrics
                    layer_diffs_per_timestep_normalized.append((l2_norm / (l2_norm_baseline + 1e-8)).item())
                    cosine_sims_per_timestep_normalized.append((cosine_sim / (sim_baseline + 1e-8)).item())
                    
                    # 3. Subtraction-normalized metrics
                    layer_diffs_per_timestep_subtracted.append((l2_norm - l2_norm_baseline).item())
                    cosine_sims_per_timestep_subtracted.append((cosine_sim - sim_baseline).item())

                # Append timestep data to prompt-level lists
                layer_diff_for_prompt.append(layer_diffs_per_timestep)
                cosine_sims_for_prompt.append(cosine_sims_per_timestep)
                layer_diff_for_prompt_normalized.append(layer_diffs_per_timestep_normalized)
                cosine_sims_for_prompt_normalized.append(cosine_sims_per_timestep_normalized)
                layer_diff_for_prompt_subtracted.append(layer_diffs_per_timestep_subtracted)
                cosine_sims_for_prompt_subtracted.append(cosine_sims_per_timestep_subtracted)

            # Append prompt-level data to global lists
            layer_diffs.append(layer_diff_for_prompt)
            cosine_sims.append(cosine_sims_for_prompt) 
            layer_diffs_normalized.append(layer_diff_for_prompt_normalized)
            cosine_sims_normalized.append(cosine_sims_for_prompt_normalized)
            layer_diffs_subtracted.append(layer_diff_for_prompt_subtracted)
            cosine_sims_subtracted.append(cosine_sims_for_prompt_subtracted)

            flush()
        
    results = {
        "layer_diffs": layer_diffs,
        "cosine_sims": cosine_sims,
        "layer_diffs_normalized": layer_diffs_normalized,
        "cosine_sims_normalized": cosine_sims_normalized,
        "layer_diffs_subtracted": layer_diffs_subtracted,
        "cosine_sims_subtracted": cosine_sims_subtracted
    }

    filename = f"MI_{args.model}_{args.target}_{args.max_prompts}_prompts.json"
    save_path = os.path.join(args.output_path, filename)
    os.makedirs(args.output_path, exist_ok=True)

    # Save the results dictionary to the specified output path
    print(f"\nSaving results to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', 'aesthetics")
    parser.add_argument("--model", type=str, required=True, help="Model type. 'flux' or 'sd3' or 'pixart'.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output *directory* to save results")

    parser.add_argument("--num_steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds per prompt.")
    parser.add_argument("--dataset_path", type=str, default='', help="Path to the JSON prompt file")
    parser.add_argument("--max_prompts", type=int, default=250)

    args = parser.parse_args()
    main(args)