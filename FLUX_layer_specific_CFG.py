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
from pathlib import Path
import torch
from datetime import datetime

from FLUX_pipeline_for_cfg import CustomFluxPipeline
from FLUX_custom_pipeline import FluxPipeline

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), f"flux_skip_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), datasets, "failure_case_prompts.json")
dataset = json.load(open(dataset_path, "r"))

# List of guidance scales to test
GUIDANCE_SCALES = [0., 0.5, 1.0, 3.0, 5.0, 7.5, 10.0]
SKIPPED_LAYERS = [5,8]

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
    subdir = os.path.join(results_dir, prompt[:10])
    os.mkdir(subdir)
    for gs in GUIDANCE_SCALES:
        print(f"Generating image with CFG scale: {gs}")

        # Run the pipeline with specified CFG
        with torch.no_grad():
            image = pipe(
                prompt,
                guidance_scale=gs,
                num_inference_steps=30,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

        # Save image
        image_path = os.path.join(subdir, f"cfgscale_{gs:.1f}.png")
        image.save(image_path)
        del image
        flush()

flush()