import os
import json
import random
import torch
import inspect
import gc
from datetime import datetime

from FLUX_custom_pipeline import FluxPipeline
from diffusers import StableDiffusion3Pipeline as SD3Pipeline

SKIPPED_LAYERS = [[9], [12], [9, 12]]
SEEDS = [0, 42, 123, 456, 789, ]  # Mehrere Seeds

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'SD3_final_experiments', f"sd3_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', "complex_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# Load model
pipe = SD3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
device = torch.device("cuda")


def flush():
    gc.collect()
    torch.cuda.empty_cache()


# Nested Loop: Prompt -> Seed -> Image Types
for prompt in dataset:
    print(f"Generating images for prompt: {prompt}")
    prompt_dir = os.path.join(results_dir, prompt)
    os.makedirs(prompt_dir, exist_ok=True)

    for seed in SEEDS:
        print(f"  Processing seed: {seed}")
        seed_dir = os.path.join(prompt_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # Generator für diesen Seed
        generator = torch.Generator("cpu").manual_seed(seed)

        # 1. No Guidance
        image_no_guidance = pipe(
            prompt,
            guidance_scale=1,
            generator=generator,
            negative_prompt="",
        ).images[0]
        no_guidance_path = os.path.join(seed_dir, "no_guidance.png")
        image_no_guidance.save(no_guidance_path)
        del image_no_guidance

        # 2. Default CFG
        generator = torch.Generator("cpu").manual_seed(seed)  # Reset generator
        image_cfg = pipe(
            prompt,
            generator=generator
        ).images[0]
        cfg_path = os.path.join(seed_dir, "default_cfg.png")
        image_cfg.save(cfg_path)
        del image_cfg

        # 3. Skip Layer Guidance
        for skipped_layers in SKIPPED_LAYERS:
            generator = torch.Generator("cpu").manual_seed(seed)  # Reset generator

            with torch.no_grad():
                image_slg = pipe(
                    prompt,
                    skip_guidance_layers=skipped_layers,
                    generator=generator
                ).images[0]

                slg_filename = f"slg_skiplayer_{'_'.join(map(str, skipped_layers))}.png"
                slg_path = os.path.join(seed_dir, slg_filename)
                image_slg.save(slg_path)
                del image_slg

        flush()

flush()