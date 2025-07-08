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

SKIPPED_LAYER = [5]
GUIDANCE_SCALE = 2.

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'final_experiments', f"flux_results_only_slg{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', "slg_vs_cfg_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# Load model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
device = torch.device("cuda")  
pipe.skipped_layers=SKIPPED_LAYER

# Cleanup 
def flush():
    gc.collect()
    torch.cuda.empty_cache()

for prompt in dataset:
    print(f"Generating images for prompt: {prompt}")
    subdir = os.path.join(results_dir, prompt[:20])
    os.mkdir(subdir)
    with torch.no_grad():
        for i in range(10):
            print(f"Generating images for random seed {i}")
            random_seed = random.randint(0, 2**32 - 1)
            seed_dir = os.path.join(subdir, f'seed_{str(i)}')
            os.makedirs(seed_dir, exist_ok=True)

            # Use no guidance at all 
            image_no_guidance = pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=28,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random_seed)
            ).images[0]
            # Save image
            no_guidance_path = os.path.join(seed_dir, f"no_guidance.png")
            image_no_guidance.save(no_guidance_path)
            del image_no_guidance

            # Use guidance for standard CFG
            image_cfg = pipe(
                prompt,
                num_inference_steps=28,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random_seed)
            ).images[0]
            # Save image
            cfg_path = os.path.join(seed_dir, f"default_cfg.png")
            image_cfg.save(cfg_path)
            print(f'Saved cfg image to: {cfg_path}')
            del image_cfg

            # Use true cfg for skip layer guidance 
            image_slg = pipe(
                prompt,
                negative_prompt=prompt,
                guidance_scale=0.0, 
                true_cfg_scale=GUIDANCE_SCALE,
                num_inference_steps=28,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(random_seed)
            ).images[0]
            slg_path = os.path.join(seed_dir, f"slg_skiplayer_{str(SKIPPED_LAYER[0])}_scale{str(GUIDANCE_SCALE)}.png")
            image_slg.save(slg_path)
            print(f'Saved slg image to: {slg_path}')
            del image_slg
                
            flush()
flush()