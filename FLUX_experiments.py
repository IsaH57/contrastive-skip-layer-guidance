import os
import json
import random
import torch
import inspect
import gc
from datetime import datetime

from FLUX_custom_pipeline import FluxPipeline

SKIPPED_LAYER = [5]
CFG_GUIDANCE_SCALES = [0., 1.5, 3., 4.5, 6.]
SLG_GUIDANCE_SCALES = [0., 1.0, 2.0, 3.0]

# Determine script name for folder naming
current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.getcwd(), 'experiments', f"flux_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Load dataset of prompt pairs 
dataset_path = os.path.join(os.getcwd(), 'prompt_datasets', 'text', "complex_prompts_2.json")
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
        for i in range(5):
            print(f"Generating images for random seed {i}")
            random_seed = random.randint(0, 2**32 - 1)
            seed_dir = os.path.join(subdir, f'seed_{str(i)}')
            os.makedirs(seed_dir, exist_ok=True)

            for slg_scale in SLG_GUIDANCE_SCALES: 
                for cfg_scale in CFG_GUIDANCE_SCALES:
                    # Use true cfg for skip layer guidance 
                    image = pipe(
                        prompt,
                        negative_prompt=prompt,
                        guidance_scale=cfg_scale, 
                        true_cfg_scale=slg_scale,
                        num_inference_steps=28,
                        max_sequence_length=256,
                        generator=torch.Generator("cpu").manual_seed(random_seed)
                    ).images[0]
                    image_path = os.path.join(seed_dir, f"slg_{str(slg_scale)}_cfg_{str(cfg_scale)}.png")
                    image.save(image_path)
                    print(f'Saved slg image to: {image_path}')
                    del image
                    flush()
flush()