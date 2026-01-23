import json
import torch
import gc
import sys
import os
import random
import torch.nn.functional as F
import argparse
import config
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from eval.eval_hands import main as evaluate_hands
from eval.eval_aesthetics import main as evaluate_aesthetics
from eval.eval_text import main as evaluate_text
from src.SD3_custom_pipeline import StableDiffusion3Pipeline
from src.FLUX_custom_pipeline import FluxPipeline
from src.PixartAlpha_custom_pipeline import PixArtAlphaPipeline

def flush():
    """Helper function to free up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def _save_geneval_leaf(root, idx, metadata_obj, pil_image):
    """
    Create a Geneval-compliant leaf:
      <root>/<idx:05d>/
        metadata.jsonl
        grid.png
        samples/0000.png
    """
    leaf = os.path.join(root, f"{idx:05d}")
    samples_dir = os.path.join(leaf, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Write one-line JSONL with the metadata for this prompt
    with open(os.path.join(leaf, "metadata.jsonl"), "w") as f:
        f.write(json.dumps(metadata_obj) + "\n")

    # Save the single sample and a "grid.png" (same image – minimal & valid)
    pil_image.save(os.path.join(samples_dir, "0000.png"))
    pil_image.save(os.path.join(leaf, "grid.png"))
    return leaf  # for logging if needed

def main(args):
    print(f'PERFORMING MSG GUIDANCE SWEEP FOR TARGET: {args.target}')
    print(f'USING MODEL: {args.model}')
    print(f'MAX PROMPTS: {args.max_prompts}')

    # Load Dataset
    match args.target:
        case 'text_and_hands':
            dataset = json.load(open('prompt_datasets/mixed_datasets/text_hands.json', "r"))
        case 'text_and_aesthetics':
            dataset = json.load(open('prompt_datasets/mixed_datasets/text_aesthetics.json', "r"))         
        case 'hands_and_aesthetics':
            dataset = json.load(open('prompt_datasets/mixed_datasets/hands_aesthetics.json', "r"))
        case 'complex_text_and_aesthetics':
            dataset = json.load(open('prompt_datasets/mixed_datasets/complex_text_aesthetics.json', "r"))
        case 'complex_text_and_hands':
            dataset = json.load(open('prompt_datasets/mixed_datasets/complex_text_hands.json', "r"))
        case _:
            if args.dataset_path != '':
                dataset = json.load(open(args.dataset_path, "r"))
            else:
                print("No valid target or dataset_path provided.")
                raise ValueError("Invalid dataset target or path")
    random.shuffle(dataset)

    # Process dataset slice
    prompts_to_process = dataset[:args.max_prompts]
    num_prompts = len(prompts_to_process)

    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.model}_{args.target}_msg_guidance_ablation_{timestamp}"
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
        case 'sd35':
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
        case _:
            raise ValueError(f"Unknown model type: {args.model}")
    pipe.to("cuda")
    print("Model loaded.")

    is_geneval = args.target in {'binding', 'counting', 'position'}

    try:
        for idx, data in enumerate(prompts_to_process):
            positive_prompt = data['positive'] if not is_geneval else data['prompt']
            
            print(f"\n--- Processing Prompt {idx+1}/{num_prompts} ---")
            print(f"Prompt: {positive_prompt}")

            # Use one seed for all variations of this prompt for consistent comparison
            seed = random.randint(0, 2**32 - 1)

            if is_geneval:
                # --- Geneval branch: separate subfolders per config (cfg, msg_weighted, etc.) ---
                
                # 1) GENERATE IMAGE WITH CFG (DEFAULT GUIDANCE SCALE)
                print("Generating for: cfg")
                pipe.skipped_layers = []
                pipe.layer_weights = []
                pipe.multiskip = False
                pipe.cfg_skip = False
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                
                cfg_root = os.path.join(run_output_path, "cfg")
                os.makedirs(cfg_root, exist_ok=True)
                leaf = _save_geneval_leaf(cfg_root, idx, data, image)
                results_log[f"cfg/{os.path.basename(leaf)}"] = positive_prompt
                flush()

                # 2) GENERATE IMAGES WITH MSG AND TARGET SPECIFIC WEIGHTS
                print("Generating for: msg_weighted")
                pipe.multiskip = True
                pipe.cfg_skip = False
                match args.model:
                    case 'flux':
                        pipe.skipped_layers = config.FLUX_LAYERS[args.target]
                        pipe.layer_weights = config.FLUX_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        pipe.skipped_layers = config.SD3_LAYERS[args.target]
                        pipe.layer_weights = config.SD3_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=config.SD3_LAYERS[args.target],
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        pipe.skipped_layers = config.PIXART_LAYERS[args.target]
                        pipe.layer_weights = config.PIXART_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale = config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        pipe.skipped_layers = config.SD35_LAYERS[args.target]
                        pipe.layer_weights = config.SD35_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=config.SD35_LAYERS[args.target],
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                
                weighted_root = os.path.join(run_output_path, "msg_weighted")
                os.makedirs(weighted_root, exist_ok=True)
                leaf = _save_geneval_leaf(weighted_root, idx, data, image)
                results_log[f"msg_weighted/{os.path.basename(leaf)}"] = positive_prompt
                flush()


            else:
                # --- Original branch: one subfolder per prompt, all configs inside ---
                prompt_subdir_name = f"{idx:05d}" # Use 5-digit padding
                prompt_output_path = os.path.join(run_output_path, prompt_subdir_name)
                os.makedirs(prompt_output_path, exist_ok=True)

                # GENERATE IMAGE WITH CFG (DEFAULT GUIDANCE SCALE)
                print("Generating for: cfg")
                pipe.skipped_layers = []
                pipe.layer_weights = []
                pipe.multiskip = False
                pipe.cfg_skip = False
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]

                # --- Save Image ---
                image_name = f"cfg.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)
                flush()

                # GENERATE IMAGES WITH MSG AND TARGET SPECIFIC WEIGHTS
                print("Generating for: msg_weighted")
                pipe.multiskip = True
                pipe.cfg_skip = False
                match args.model:
                    case 'flux':
                        pipe.skipped_layers = config.FLUX_LAYERS[args.target]
                        pipe.layer_weights = config.FLUX_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        pipe.skipped_layers = config.SD3_LAYERS[args.target]
                        pipe.layer_weights = config.SD3_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=config.SD3_LAYERS[args.target],
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        pipe.skipped_layers = config.PIXART_LAYERS[args.target]
                        pipe.layer_weights = config.PIXART_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale = config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        pipe.skipped_layers = config.SD35_LAYERS[args.target]
                        pipe.layer_weights = config.SD35_WEIGHTS[args.target]
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=config.SD35_LAYERS[args.target],
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                
                # --- Save Image ---
                image_name = f"msg_weighted.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)
                flush()

              
                # --- Log to JSON ---
                results_log[prompt_subdir_name] = positive_prompt

    finally:
        # --- Save JSON Log ---
        json_save_path = os.path.join(run_output_path, "results_log.json")
        print(f"\nSaving results log to: {json_save_path}")
        with open(json_save_path, "w") as f:
            json.dump(results_log, f, indent=4)

        print(f"\n--- Starting Automatic Evaluation for target: {args.target} ---")
        args.path = run_output_path
        args.detection_confidence = 0.7  # Default confidence for hand evaluation
        
        # Updated evaluation block to correctly handle all cases
        match args.target:
            case 'text_and_hands':
                evaluate_hands(args)
                evaluate_text(args)
            case 'text_and_aesthetics':
                evaluate_aesthetics(args)
                evaluate_text(args)
            case 'complex_text_aesthetics':
                evaluate_aesthetics(args)
                evaluate_text(args)
            case 'hands_and_aesthetics':
                evaluate_hands(args)
                evaluate_aesthetics(args)
            case _:
                print(f"No automatic evaluation script found for target: {args.target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', 'aesthetics'.")
    parser.add_argument("--model", type=str, required=True, help="Model type. 'flux', 'sd3', or 'pixart'.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output *directory* to save results.")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to a custom JSON dataset if target is not standard.")
    parser.add_argument("--max_prompts", type=int, default=100, help="Maximum number of prompts to process from the dataset.")

    args = parser.parse_args()
    main(args)