import os
import json
import random
import torch
import inspect
import gc
import argparse
from datetime import datetime
from diffusers import StableDiffusion3Pipeline as SD3Pipeline

from FLUX_custom_pipeline import FluxPipeline

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    SKIPPED_LAYERS = [i for i in range(19)]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), 'experiments', f"{args.model}_layer_ablation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset of prompts
    dataset = json.load(open(args.dataset_path, "r"))

    # Load model
    if args.model == 'flux':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    elif args.model == 'sd3':
        pipe = SD3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    else:
        raise NameError('Invalid model name.')

    for pair in dataset[33:]:
        print(f"Generating images for prompt: {pair['positive']}")
        subdir = os.path.join(results_dir, pair['positive'][:20])
        os.mkdir(subdir)
        with torch.no_grad():
            for i in range(args.num_seeds):
                print(f"Generating images for random seed {i}")
                random_seed = random.randint(0, 2**32 - 1)
                seed_dir = os.path.join(subdir, f'seed_{str(i)}')
                os.makedirs(seed_dir, exist_ok=True)
                if args.model == 'flux':
                    pipe.skipped_layers = []
                    image = pipe(
                        prompt=pair['positive'],
                        max_sequence_length=256,
                        generator=torch.Generator("cpu").manual_seed(random_seed)
                    ).images[0]
                elif args.model == 'sd3':
                    image = pipe(
                        pair['positive'],
                        generator=torch.Generator("cpu").manual_seed(random_seed)
                    ).images[0]
                    
                image_path = os.path.join(seed_dir, f"cfg.png")
                image.save(image_path)
                print(f'Saved slg image to: {image_path}')
                del image
                flush()


                for layer in SKIPPED_LAYERS:
                    # Use true cfg for skip layer guidance 
                    if args.model == 'flux':
                        pipe.skipped_layers=[layer]
                        image = pipe(
                            prompt=pair['positive'],
                            negative_prompt=pair['positive'],
                            true_cfg_scale=0.,
                            max_sequence_length=256,
                            generator=torch.Generator("cpu").manual_seed(random_seed)
                        ).images[0]
                    elif args.model == 'sd3':
                        image = pipe(
                            pair['positive'],
                            skip_guidance_layers=[layer],
                            skip_layer_guidance_scale=0.,
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cpu").manual_seed(random_seed)
                        ).images[0]
                    
                    image_path = os.path.join(seed_dir, f"layer_{str(layer)}.png")
                    image.save(image_path)
                    print(f'Saved slg image to: {image_path}')
                    del image
                    flush()
        flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")

    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds")

    args = parser.parse_args()
    main(args)