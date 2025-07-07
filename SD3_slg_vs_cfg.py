import os
import json
import random
import torch
import inspect
import os
import torch
import gc
import inspect
import os
import torch
from datetime import datetime

from FLUX_custom_pipeline import FluxPipeline
from diffusers import StableDiffusion3Pipeline as SD3Pipeline

SKIPPED_LAYERS = [[9], [12],[9,12]]
GUIDANCE_SCALES = [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # 1.0 means no guidance, 7.0 is the default CFG scale

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'SD3_CFG_vs_SLG_experiments', f"sd3_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', "test_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# Load model
pipe = SD3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
device = torch.device("cuda")


# Cleanup
def flush():
    gc.collect()
    torch.cuda.empty_cache()


for prompt in dataset:
    print(f"Generating images for prompt: {prompt}")
    subdir = os.path.join(results_dir, prompt)
    os.mkdir(subdir)

    # Use no guidance at all
    image_no_guidance = pipe(
        prompt,
        guidance_scale=1, # corresponds to no guidance
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    # Save image
    no_guidance_path = os.path.join(subdir, f"no_guidance.png")
    image_no_guidance.save(no_guidance_path)
    del image_no_guidance

    # Use guidance for standard CFG
    image_cfg = pipe(
        prompt,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    # Save image
    cfg_path = os.path.join(subdir, f"default_cfg.png")
    image_cfg.save(cfg_path)
    print(f'Saved cfg image to: {cfg_path}')
    del image_cfg

    for skipped_layers in SKIPPED_LAYERS:
        for gs in GUIDANCE_SCALES:
            #pipe.skipped_layers = skipped_layers

            # Run the pipeline with specified CFG
            with torch.no_grad():
                image_slg = pipe(
                    prompt,
                    guidance_scale=gs,
                    skip_guidance_layers=skipped_layers,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
                slg_path = os.path.join(subdir, f"slg_no_negprompt_scale_{gs}_layers" + '_'.join(
                    map(str, skipped_layers)) + ".png")
                image_slg.save(slg_path)
                print(f'Saved slg image to: {slg_path}')
                del image_slg

        flush()
flush()