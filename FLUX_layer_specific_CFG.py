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

# List of guidance scales to test
GUIDANCE_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
SKIPPED_LAYERS = [8]

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'CFG_VS_SLG_experiments', f"flux_results_{timestamp}_layers")
for layer in SKIPPED_LAYERS:
    results_dir = results_dir + "_" + str(layer)
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', "slg_vs_cfg_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# Load model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.skipped_layers=SKIPPED_LAYERS
pipe.enable_model_cpu_offload()
device = torch.device("cuda")  

# Cleanup 
def flush():
    gc.collect()
    torch.cuda.empty_cache()

for prompt in dataset:
    subdir = os.path.join(results_dir, prompt[:12])
    os.mkdir(subdir)
    print(f"Generating images for prompt: {prompt}")
    for gs in GUIDANCE_SCALES:
        # Run the pipeline with specified CFG
        with torch.no_grad():
            # Use true cfg for skip layer guidance 
            if gs != 1.:
                image_slg = pipe(
                    prompt,
                    negative_prompt=prompt,
                    true_cfg_scale=gs,
                    num_inference_steps=28,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
                slg_path = os.path.join(subdir, f"slg_scale_{gs:.1f}_layers" + '_'.join(map(str, SKIPPED_LAYERS)) + ".png")
                image_slg.save(slg_path)
                print(f'Saved slg image to: {slg_path}')
                del image_slg

            # Use guidance for standard CFG
            image_cfg = pipe(
                prompt,
                guidance_scale=gs,
                num_inference_steps=28,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

        # Save image
        if gs != 1.: 
            cfg_path = os.path.join(subdir, f"cfg_scale_{gs:.1f}.png")
            image_cfg.save(cfg_path)
            print(f'Saved cfg image to: {cfg_path}')
        else: 
            no_guidance_path = os.path.join(subdir, "no_guidance.png")
            image_cfg.save(no_guidance_path)
        del image_cfg
    
        flush()

flush()