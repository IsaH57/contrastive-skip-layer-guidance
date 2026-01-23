import json
import torch
import gc
import sys
import os
import random
import argparse
import config
from datetime import datetime

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.SD3_custom_pipeline import StableDiffusion3Pipeline
from src.FLUX_custom_pipeline import FluxPipeline 
from src.PixartAlpha_custom_pipeline import PixArtAlphaPipeline

def flush():
    """Helper to free up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def get_config_layers_and_weights(model_type, target, limit_layers=None):
    """Retrieves layers/weights and slices them if limit_layers is set."""
    if model_type == 'flux':
        layers = config.FLUX_LAYERS[target]
        weights = config.FLUX_WEIGHTS[target]
    elif model_type == 'sd3':
        layers = config.SD3_LAYERS[target]
        weights = config.SD3_WEIGHTS[target]
    elif model_type == 'sd35':
        layers = config.SD35_LAYERS[target]
        weights = config.SD35_WEIGHTS[target]
    elif model_type == 'pixart':
        layers = config.PIXART_LAYERS[target]
        weights = config.PIXART_WEIGHTS[target]
    else:
        raise ValueError(f"Unknown model: {model_type}")

    if limit_layers is not None and limit_layers > 0:
        layers = layers[:limit_layers]
        weights = weights[:limit_layers]
        print(f"--- Limiting to top {limit_layers} skipped layers ---")
    else:
        print(f"--- Using all {len(layers)} defined skipped layers ---")

    return layers, weights

def load_dataset(args):
    """Loads the specific dataset based on target."""
    print(f"Loading dataset for target: {args.target}...")
    dataset = []
    
    match args.target: 
        case 'text': 
            dataset = json.load(open('prompt_datasets/text_pairs.json', "r"))
        case 'complex_text':
            dataset = json.load(open('prompt_datasets/text/complex_prompt_pairs.json', "r"))
        case 'hands':
            dataset = json.load(open('prompt_datasets/hands_pairs.json', "r"))
        case 'aesthetics':
            dataset = json.load(open('prompt_datasets/aesthetics_pairs.json', "r"))
        case 'binding':
            with open('prompt_datasets/geneval/attribute_binding_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'counting':
            with open('prompt_datasets/geneval/counting_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'position':
            with open('prompt_datasets/geneval/position_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case _: 
            if args.dataset_path != '':
                dataset = json.load(open(args.dataset_path, "r"))
            else:
                raise ValueError("Invalid target or no dataset_path provided.")
    
    print(f"Total prompts in dataset: {len(dataset)}")
    random.shuffle(dataset)
    
    # Select N random prompts
    selected = dataset[:args.num_prompts]
    print(f"Selected {len(selected)} prompts for processing.")
    return selected

def main(args):
    # --- Setup Output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"seed_sweep_{args.model}_{args.target}_{timestamp}"
    run_output_path = os.path.join(args.output_path, run_dir_name)
    os.makedirs(run_output_path, exist_ok=True)

    # --- Load Dataset ---
    prompts_data = load_dataset(args)
    
    # --- Load Model ---
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

    # --- Prepare Layers ---
    active_layers, active_weights = get_config_layers_and_weights(args.model, args.target, args.limit_layers)

    # Determine if geneval (uses 'prompt' key) or standard (uses 'positive' key)
    is_geneval = args.target in {'binding', 'counting', 'position'}

    results_log = {}

    # --- Main Processing Loop ---
    for p_idx, data in enumerate(prompts_data):
        prompt_text = data['prompt'] if is_geneval else data['positive']
        
        # Create folder structure: /0000/cfg and /0000/msg
        prompt_base_folder = os.path.join(run_output_path, f"{p_idx:04d}")
        cfg_folder = os.path.join(prompt_base_folder, "cfg")
        msg_folder = os.path.join(prompt_base_folder, "msg")

        os.makedirs(cfg_folder, exist_ok=True)
        os.makedirs(msg_folder, exist_ok=True)

        print(f"\n=== Prompt {p_idx+1}/{len(prompts_data)} ===")
        print(f"Text: {prompt_text[:60]}...")

        # Save metadata in the base numbered folder
        with open(os.path.join(prompt_base_folder, "metadata.json"), "w") as f:
            json.dump(data, f, indent=4)
        
        results_log[f"{p_idx:04d}"] = prompt_text

        # --- Seed Loop ---
        for s_idx in range(args.num_seeds):
            seed = random.randint(0, 2**32 - 1)
            print(f"   > Seed {s_idx+1}/{args.num_seeds}: {seed}")

            # 1. Generate Baseline (CFG) -> Save to /cfg/
            match args.model:
                case 'flux':
                    pipe.skipped_layers = []   
                    image = pipe(prompt=prompt_text, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                case 'sd3' | 'sd35':
                    pipe.multiskip = False
                    pipe.cfg_skip = False
                    image = pipe(prompt=prompt_text, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                case 'pixart':
                    pipe.multiskip = False
                    pipe.cfg_skip = False
                    pipe.skipped_layers = []
                    image = pipe(prompt=prompt_text, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            
            # Save to CFG folder
            image.save(os.path.join(cfg_folder, f"seed_{seed}.jpg"))
            flush()

            # 2. Generate MSG -> Save to /msg/
            match args.model:
                case 'flux':
                    pipe.multiskip = True
                    pipe.skipped_layers = active_layers
                    pipe.layer_weights = active_weights
                    image = pipe(
                        prompt=prompt_text, 
                        negative_prompt=prompt_text,    
                        true_cfg_scale=args.msg_scale,                 
                        generator=torch.Generator("cuda").manual_seed(seed)
                    ).images[0]
                case 'sd3':
                    pipe.multiskip = True
                    pipe.cfg_skip = False
                    pipe.skipped_layers = active_layers
                    pipe.layer_weights = active_weights
                    image = pipe(
                        prompt=prompt_text, 
                        skip_guidance_layers=active_layers,
                        skip_layer_guidance_scale=args.msg_scale,
                        skip_layer_guidance_start=0.,
                        skip_layer_guidance_stop=1., 
                        generator=torch.Generator("cuda").manual_seed(seed)
                    ).images[0]
                case 'pixart':
                    pipe.multiskip = True
                    pipe.cfg_skip = False 
                    pipe.skipped_layers = active_layers
                    pipe.layer_weights = active_weights
                    image = pipe(
                        prompt=prompt_text, 
                        skip_layer_guidance_scale = args.msg_scale,
                        generator=torch.Generator("cuda").manual_seed(seed)
                    ).images[0]
                case 'sd35':
                    pipe.multiskip = True
                    pipe.cfg_skip = False
                    pipe.skipped_layers = active_layers
                    pipe.layer_weights = active_weights
                    image = pipe(
                        prompt=prompt_text, 
                        skip_guidance_layers=active_layers,
                        skip_layer_guidance_scale=args.msg_scale,
                        skip_layer_guidance_start=0.,
                        skip_layer_guidance_stop=1., 
                        generator=torch.Generator("cuda").manual_seed(seed)
                    ).images[0]

            # Save to MSG folder
            image.save(os.path.join(msg_folder, f"seed_{seed}.jpg"))
            flush()

    # Save global log
    with open(os.path.join(run_output_path, "prompts_log.json"), "w") as f:
        json.dump(results_log, f, indent=4)
    
    print(f"\nDone. Results saved to: {run_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run seed sweep on random prompts from dataset.")
    
    parser.add_argument("--model", type=str, required=True, choices=['flux', 'sd3', 'sd35', 'pixart'])
    parser.add_argument("--target", type=str, required=True, help="Target (hands, text, etc).")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="", help="Custom path if target is not standard.")
    
    # Controls
    parser.add_argument("--num_prompts", type=int, default=5, help="Prompts to select.")
    parser.add_argument("--num_seeds", type=int, default=3, help="Seeds per prompt.")
    parser.add_argument("--msg_scale", type=float, default=2.0, help="Fixed MSG guidance scale.")
    parser.add_argument("--limit_layers", type=int, default=0, help="Limit number of skipped layers.")

    args = parser.parse_args()
    main(args)