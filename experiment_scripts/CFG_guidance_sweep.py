import json
import torch
import gc
import sys
import os
import random
import torch.nn.functional as F
import argparse
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline 
from diffusers import PixArtAlphaPipeline
from datetime import datetime 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def flush():
    """Helper function to free up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def main(args):

    print(f'PERFORMING CFG SWEEP FOR TARGET: {args.target}')
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
                print("No valid target or dataset_path provided.")
                raise ValueError("Invalid dataset target or path")
    random.shuffle(dataset)

    # --- METRIC LISTS ---
    cfg_scales = [float(i/2) for i in range(2, 22)]
    
    # Process dataset slice
    prompts_to_process = dataset[:args.max_prompts]
    num_prompts = len(prompts_to_process)
    num_images_total = num_prompts * len(cfg_scales)

    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.model}_{args.target}_{num_images_total}_{timestamp}"
    run_output_path = os.path.join(args.output_path, run_dir_name)
    os.makedirs(run_output_path, exist_ok=True)
    print(f"Saving results to: {run_output_path}")

    # --- JSON Log ---
    # This dictionary stores {relative_image_path: prompt_string} pairs
    results_log = {}

    # Load Pipeline
    print(f"Loading model: {args.model}...")
    match args.model: 
        case 'flux':
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        case 'sd3':
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        case 'pixart':
            pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        case _:
            raise ValueError(f"Unknown model type: {args.model}")
    pipe.to("cuda")
    print("Model loaded.")

    try:
        for idx, prompt_pair in enumerate(prompts_to_process):
            positive_prompt = prompt_pair['positive']
            print(f"\n--- Processing Prompt {idx+1}/{num_prompts} ---")
            print(f"Prompt: {positive_prompt}")

            # Create prompt-specific subfolder
            prompt_subdir_name = f"prompt_{idx}"
            prompt_output_path = os.path.join(run_output_path, prompt_subdir_name)
            os.makedirs(prompt_output_path, exist_ok=True)

            # Use one seed for all CFG scales of this prompt for consistent comparison
            seed = random.randint(0, 2**32 - 1)

            for cfg in cfg_scales:
                print(f"  Generating with CFG: {cfg:.1f}")

                # Set generator with the fixed seed for this prompt
                generator = torch.Generator("cuda").manual_seed(seed)
                
                # Generate the image
                image = pipe(
                    prompt=positive_prompt, 
                    guidance_scale=cfg, 
                    generator=generator
                ).images[0]

                # --- Save Image ---
                image_name = f"cfg_scale_{cfg:.1f}.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)

                # --- Log to JSON ---
                relative_image_path = os.path.join(prompt_subdir_name, image_name)
                results_log[relative_image_path] = positive_prompt

            flush()

    finally:
        # --- Save JSON Log ---
        json_save_path = os.path.join(run_output_path, "results_log.json")
        print(f"\nSaving results log to: {json_save_path}")
        with open(json_save_path, "w") as f:
            json.dump(results_log, f, indent=4)

        print(f"--- Run complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', 'aesthetics'.")
    parser.add_argument("--model", type=str, required=True, help="Model type. 'flux', 'sd3', or 'pixart'.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output *directory* to save results.")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to a custom JSON dataset if target is not standard.")
    parser.add_argument("--max_prompts", type=int, default=100, help="Maximum number of prompts to process from the dataset.")

    args = parser.parse_args()
    main(args)