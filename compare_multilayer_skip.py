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
    if args.target == 'text': 
        SKIPPED_LAYERS = [8, 17, 18, 6, 14]
        LAYER_WEIGHTS = [1., 0.6014, 0.5911, 0.5815, 0.5744]
        dataset_path = '/export/home/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/text/complex_prompts.json'
    elif args.target == 'aesthetics':
        SKIPPED_LAYERS = [15, 14, 8, 12, 10]
        LAYER_WEIGHTS = [1., 0.9381, 0.8042, 0.7674, 0.7545]
        dataset_path = '/export/home/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/aesthetics/aesthetics.json'
    else: 
        print('Unknown target.')
        SKIPPED_LAYERS = [int(x) for x in args.skipped_layers.split(",")] 
        LAYER_WEIGHTS = [float(x) for x in args.layer_weights.split(",")] 
    SLG_GUIDANCE_SCALES = [float(x) for x in args.slg_guidance_scales.split(",")] if args.slg_guidance_scales else [1.5, 2., 3., 4., 5.]
    print(f'Using layers {str(SKIPPED_LAYERS)}, weights {LAYER_WEIGHTS} and guidance scales {SLG_GUIDANCE_SCALES} for skipping.')

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), 'experiments', f"{args.model}_multiskip_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset of prompts
    dataset = json.load(open(dataset_path, "r"))

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

    for prompt in dataset:
        print(f"Generating images for prompt: {prompt}")
        subdir = os.path.join(results_dir, prompt[:20])
        os.mkdir(subdir)
        with torch.no_grad():
            for i in range(args.num_seeds):
                print(f"Generating images for random seed {i}")
                random_seed = random.randint(0, 2**32 - 1)
                seed_dir = os.path.join(subdir, f'seed_{str(i)}')
                os.makedirs(seed_dir, exist_ok=True)

                # No SLG
                image_no_slg = pipe(
                    prompt,
                    num_inference_steps=28,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(random_seed)
                ).images[0]
                image_path = os.path.join(seed_dir, f"cfg_only.png")
                image_no_slg.save(image_path)
                print(f'Saved slg image to: {image_path}')
                del image_no_slg
                flush()

                for slg_scale in SLG_GUIDANCE_SCALES: 
                    # Single skip
                    pipe.multiskip=False
                    pipe.skipped_layers = [SKIPPED_LAYERS[0]]
                    image = pipe(
                        prompt,
                        negative_prompt=prompt,
                        true_cfg_scale=slg_scale,
                        num_inference_steps=28,
                        max_sequence_length=256,
                        generator=torch.Generator("cpu").manual_seed(random_seed)
                    ).images[0]
                    
                    image_path = os.path.join(seed_dir, f"single_skip_slg_{str(slg_scale)}.png")
                    image.save(image_path)
                    print(f'Saved slg image to: {image_path}')
                    del image
                    flush()
                    for num_skipped_layers in [4,5]:
                        # Naive multilayer skip
                        pipe.multiskip=False
                        pipe.skipped_layers = SKIPPED_LAYERS[:num_skipped_layers]
                        image = pipe(
                            prompt,
                            negative_prompt=prompt,
                            true_cfg_scale=1 + ((slg_scale - 1) / num_skipped_layers),
                            num_inference_steps=28,
                            max_sequence_length=256,
                            generator=torch.Generator("cpu").manual_seed(random_seed)
                        ).images[0]
                        
                        image_path = os.path.join(seed_dir, f"naive_{str(num_skipped_layers)}_layers_slg_{str(slg_scale)}.png")
                        image.save(image_path)
                        print(f'Saved slg image to: {image_path}')
                        del image
                        flush()


                        '''# Multilayer Skip with summed noise over single skipped layers
                        pipe.multiskip=True
                        pipe.layer_weights=[1.,1.,1.,1.,1.]
                        image = pipe(
                            prompt,
                            negative_prompt=prompt,
                            true_cfg_scale=slg_scale,
                            num_inference_steps=28,
                            max_sequence_length=256,
                            generator=torch.Generator("cpu").manual_seed(random_seed)
                        ).images[0]
                        
                        image_path = os.path.join(seed_dir, f"fixed_{str(num_skipped_layers)}_skipped_layers_slg_{str(slg_scale)}.png")
                        image.save(image_path)
                        print(f'Saved slg image to: {image_path}')
                        del image
                        flush()'''


                        # Multilayer Skip with weighted noise over single skipped layers 
                        pipe.layer_weights = LAYER_WEIGHTS
                        image = pipe(
                            prompt,
                            negative_prompt=prompt,
                            true_cfg_scale=slg_scale,
                            num_inference_steps=28,
                            max_sequence_length=256,
                            generator=torch.Generator("cpu").manual_seed(random_seed)
                        ).images[0]
                        
                        image_path = os.path.join(seed_dir, f"weighted_{str(num_skipped_layers)}_skipped_layers_slg_{str(slg_scale)}.png")
                        image.save(image_path)
                        print(f'Saved slg image to: {image_path}')
                        del image
                        flush()
    flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")

    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--target", type=str, default="text")
    parser.add_argument("--skipped_layers", type=str, default="8,17,18,6,14", help="Comma-separated list of skipped layers")
    parser.add_argument("--slg_guidance_scales", type=str, default="1.5,2.,3.,4.,5.", help="Comma-separated SLG scales")
    parser.add_argument("--layer_weights", type=str, default="1.,0.6014,0.5911,0.5815,0.5744", help="Comma-separated layer weights")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds")

    args = parser.parse_args()
    main(args)