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
    SKIPPED_LAYERS = [int(x) for x in args.skipped_layers.split(",")] if args.skipped_layers else [5]
    CFG_GUIDANCE_SCALES = [float(x) for x in args.cfg_guidance_scales.split(",")] if args.cfg_guidance_scales else [1., 2., 3., 4., 5.]
    SLG_GUIDANCE_SCALES = [float(x) for x in args.slg_guidance_scales.split(",")] if args.slg_guidance_scales else [1., 2., 3., 4., 5.]

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), 'experiments', f"{args.model}_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset of prompts
    dataset = json.load(open(args.dataset_path, "r"))

    # Load model
    if args.model == 'flux':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipe.skipped_layers=SKIPPED_LAYERS
    elif args.model == 'sd3':
        pipe = SD3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    else:
        raise NameError('Invalid model name.')

    for prompt in dataset:
        print(f"Generating images for prompt: {prompt}")
        subdir = os.path.join(results_dir, prompt[:20])
        os.mkdir(subdir)
        with torch.no_grad():
            for i in range(5):
                print(f"Generating images for random seed {i}")
                random_seed = random.randint(0, 2**32 - 1)
                seed_generator = torch.Generator("cpu").manual_seed(random_seed)
                seed_dir = os.path.join(subdir, f'seed_{str(i)}')
                os.makedirs(seed_dir, exist_ok=True)

                for slg_scale in SLG_GUIDANCE_SCALES: 
                    for cfg_scale in CFG_GUIDANCE_SCALES:
                        # Use true cfg for skip layer guidance 
                        if args.model == 'flux':
                            image = pipe(
                                prompt,
                                negative_prompt=prompt,
                                guidance_scale=cfg_scale, 
                                true_cfg_scale=slg_scale,
                                num_inference_steps=28,
                                max_sequence_length=256,
                                generator=torch.Generator("cpu").manual_seed(random_seed)
                            ).images[0]
                        elif args.model == 'sd3':
                            image = pipe(
                                prompt,
                                skip_guidance_layers=SKIPPED_LAYERS,
                                skip_layer_guidance_scale=slg_scale,
                                skip_layer_guidance_start=0.,
                                skip_layer_guidance_stop=1.,
                                generator=torch.Generator("cpu").manual_seed(random_seed),
                                guidance_scale=cfg_scale
                            ).images[0]
                        
                        image_path = os.path.join(seed_dir, f"slg_{str(slg_scale)}_cfg_{str(cfg_scale)}.png")
                        image.save(image_path)
                        print(f'Saved slg image to: {image_path}')
                        del image
                        flush()
    flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")

    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--skipped_layers", type=str, default="5", help="Comma-separated list of skipped layers")
    parser.add_argument("--cfg_guidance_scales", type=str, default="1.,2.,3.,4.,5.", help="Comma-separated CFG scales")
    parser.add_argument("--slg_guidance_scales", type=str, default="1.,2.,3.,4.,5.", help="Comma-separated SLG scales")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds")

    args = parser.parse_args()
    main(args)