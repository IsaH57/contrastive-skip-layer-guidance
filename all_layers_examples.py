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

SKIPPED_LAYERS = [[i] for i in range(19)]
GUIDANCE_SCALES = [2.]

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'all_layers_experiments', f"flux_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', "slg_vs_cfg_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# Load model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
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
        guidance_scale=0.0,
        num_inference_steps=28,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    # Save image
    no_guidance_path = os.path.join(subdir, f"no_guidance.png")
    image_no_guidance.save(no_guidance_path)
    del image_no_guidance

    # Use guidance for standard CFG
    image_cfg = pipe(
        prompt,
        num_inference_steps=28,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    # Save image
    cfg_path = os.path.join(subdir, f"default_cfg.png")
    image_cfg.save(cfg_path)
    print(f'Saved cfg image to: {cfg_path}')
    del image_cfg

    for skipped_layers in SKIPPED_LAYERS:
        for gs in GUIDANCE_SCALES:
            pipe.skipped_layers=skipped_layers
            
            # Run the pipeline with specified CFG
            with torch.no_grad():
                # Use true cfg for skip layer guidance 

                image_slg = pipe(
                    prompt,
                    negative_prompt=prompt,
                    true_cfg_scale=gs,
                    num_inference_steps=28,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
                slg_path = os.path.join(subdir, f"slg_prompt_scale_{gs}_layers" + '_'.join(map(str, skipped_layers)) + ".png")
                image_slg.save(slg_path)
                print(f'Saved slg image to: {slg_path}')
                del image_slg
            
        flush()
flush()