import json
import torch
import gc
import sys
import os
import random
import time
import torch.nn.functional as F
import argparse
import config
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from eval.eval_hands import main as evaluate_hands
from eval.eval_aesthetics import main as evaluate_aesthetics
from eval.eval_text import main as evaluate_text
from src.SD3_custom_pipeline import StableDiffusion3Pipeline
from src.FLUX_custom_pipeline import FluxPipeline
from src.PixartAlpha_custom_pipeline import PixArtAlphaPipeline

def flush():
    """Helper function to free up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def _save_geneval_leaf(root, idx, metadata_obj, pil_image):
    """
    Create a Geneval-compliant leaf:
      <root>/<idx:05d>/
        metadata.jsonl
        grid.png
        samples/0000.png
    """
    leaf = os.path.join(root, f"{idx:05d}")
    samples_dir = os.path.join(leaf, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Write one-line JSONL with the metadata for this prompt
    with open(os.path.join(leaf, "metadata.jsonl"), "w") as f:
        f.write(json.dumps(metadata_obj) + "\n")

    # Save the single sample and a "grid.png" (same image – minimal & valid)
    pil_image.save(os.path.join(samples_dir, "0000.png"))
    pil_image.save(os.path.join(leaf, "grid.png"))
    return leaf  # for logging if needed

def main(args):
    print(f'PERFORMING MSG GUIDANCE SWEEP FOR TARGET: {args.target}')
    print(f'USING MODEL: {args.model}')
    print(f'MAX PROMPTS: {args.max_prompts}')

    # Load Dataset
    match args.target:
        case 'text':
            dataset = json.load(open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/text_pairs.json', "r"))
        case 'complex_text':
            dataset = json.load(open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/text/complex_prompt_pairs.json', "r"))
        case 'hands':
            dataset = json.load(open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/hands_pairs.json', "r"))
        case 'aesthetics':
            dataset = json.load(open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/aesthetics_pairs.json', "r"))
        case 'binding':
            with open('prompt_datasets/geneval/attribute_binding_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'counting':
            with open('prompt_datasets/geneval/counting_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'position':
            with open('prompt_datasets/geneval/position_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'text_and_hands':
            with open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/mixed_datasets/text_hands.json') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'text_and_aesthetics':
            with open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/mixed_datasets/text_aesthetics.json') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'hands_and_aesthetics':
            with open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/mixed_datasets/hands_aesthetics.json') as fp:
                dataset = [json.loads(line) for line in fp]
        case _:
            if args.dataset_path != '':
                dataset = json.load(open(args.dataset_path, "r"))
            else:
                print("No valid target or dataset_path provided.")
                raise ValueError("Invalid dataset target or path")
    random.shuffle(dataset)

    # Process dataset slice
    prompts_to_process = dataset[:args.max_prompts]
    num_prompts = len(prompts_to_process)

    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.model}_{args.target}_msg_guidance_ablation_{timestamp}"
    run_output_path = os.path.join(args.output_path, run_dir_name)
    os.makedirs(run_output_path, exist_ok=True)
    print(f"Saving results to: {run_output_path}")

    # --- JSON Log ---
    # This dictionary stores {relative_image_path: prompt_string} pairs
    results_log = {}

    # Load Pipeline
    print(f"Loading model: {args.model}...")
    match args.model:
        case 'flux':
            pipe = FluxPipeline.from_pretrained("/export/scratch/ru63zus/hub/flux", torch_dtype=torch.float16)
        case 'sd3':
            pipe = StableDiffusion3Pipeline.from_pretrained("/export/scratch/ru63zus/hub/sd3-diffusers", torch_dtype=torch.float16)
        case 'pixart':
            pipe = PixArtAlphaPipeline.from_pretrained("/export/scratch/ru63zus/hub/pixart-alpha", torch_dtype=torch.float16)
        case 'sd35':
            pipe = StableDiffusion3Pipeline.from_pretrained("/export/scratch/ru63zus/hub/sd3.5-medium", torch_dtype=torch.bfloat16)
        case _:
            raise ValueError(f"Unknown model type: {args.model}")
    pipe.to("cuda")
    print("Model loaded.")

    is_geneval = args.target in {'binding', 'counting', 'position'}

    # Timing variables
    time_cfg, runs_cfg = 0.0, 0
    time_stg, runs_stg = 0.0, 0
    time_msg, runs_msg = 0.0, 0

    try:
        for idx, data in enumerate(prompts_to_process):
            positive_prompt = data['positive'] if not is_geneval else data['prompt']
            
            print(f"\n--- Processing Prompt {idx+1}/{num_prompts} ---")
            print(f"Prompt: {positive_prompt}")

            # Use one seed for all variations of this prompt for consistent comparison
            seed = random.randint(0, 2**32 - 1)

            if is_geneval:
                # --- Geneval branch: separate subfolders per config ---
                
                # 1) GENERATE IMAGE WITH CFG
                print("Generating for: cfg")
                pipe.skipped_layers = []
                pipe.layer_weights = []
                pipe.multiskip = False
                pipe.cfg_skip = False
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_cfg += (time.perf_counter() - t0)
                runs_cfg += 1
                
                cfg_root = os.path.join(run_output_path, "cfg")
                os.makedirs(cfg_root, exist_ok=True)
                leaf = _save_geneval_leaf(cfg_root, idx, data, image)
                results_log[f"cfg/{os.path.basename(leaf)}"] = positive_prompt
                flush()

                # 2) GENERATE IMAGE WITH STG (1 Skipped Layer)
                print("Generating for: stg")
                pipe.multiskip = True
                pipe.cfg_skip = False
                pipe.skipped_layers = [2]
                pipe.layer_weights = [1.0]
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale=config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_stg += (time.perf_counter() - t0)
                runs_stg += 1

                stg_root = os.path.join(run_output_path, "stg")
                os.makedirs(stg_root, exist_ok=True)
                leaf = _save_geneval_leaf(stg_root, idx, data, image)
                results_log[f"stg/{os.path.basename(leaf)}"] = positive_prompt
                flush()

                # 3) GENERATE IMAGE WITH MSG (2 Skipped Layers)
                print("Generating for: msg")
                pipe.multiskip = True
                pipe.cfg_skip = False
                pipe.skipped_layers = [2, 4]
                pipe.layer_weights = [1.0, 1.0]
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale=config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_msg += (time.perf_counter() - t0)
                runs_msg += 1
                
                msg_root = os.path.join(run_output_path, "msg")
                os.makedirs(msg_root, exist_ok=True)
                leaf = _save_geneval_leaf(msg_root, idx, data, image)
                results_log[f"msg/{os.path.basename(leaf)}"] = positive_prompt
                flush()

            else:
                # --- Original branch: one subfolder per prompt, all configs inside ---
                prompt_subdir_name = f"{idx:05d}" # Use 5-digit padding
                prompt_output_path = os.path.join(run_output_path, prompt_subdir_name)
                os.makedirs(prompt_output_path, exist_ok=True)

                # 1) GENERATE IMAGE WITH CFG
                print("Generating for: cfg")
                pipe.skipped_layers = []
                pipe.layer_weights = []
                pipe.multiskip = False
                pipe.cfg_skip = False
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_cfg += (time.perf_counter() - t0)
                runs_cfg += 1

                # --- Save Image ---
                image_name = f"cfg.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)
                flush()

                # 2) GENERATE IMAGE WITH STG (1 Skipped Layer)
                print("Generating for: stg")
                pipe.multiskip = True
                pipe.cfg_skip = False
                pipe.skipped_layers = [2]
                pipe.layer_weights = [1.0]
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale=config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_stg += (time.perf_counter() - t0)
                runs_stg += 1
                
                # --- Save Image ---
                image_name = f"stg.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)
                flush()

                # 3) GENERATE IMAGE WITH MSG (2 Skipped Layers)
                print("Generating for: msg")
                pipe.multiskip = True
                pipe.cfg_skip = False
                pipe.skipped_layers = [2, 4]
                pipe.layer_weights = [1.0, 1.0]
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                match args.model:
                    case 'flux':
                        image = pipe(
                            prompt=positive_prompt,
                            negative_prompt=positive_prompt,
                            true_cfg_scale=config.FLUX_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd3':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD3_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'pixart':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_layer_guidance_scale=config.PIXART_SCALES[args.target],
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    case 'sd35':
                        image = pipe(
                            prompt=positive_prompt,
                            skip_guidance_layers=pipe.skipped_layers,
                            skip_layer_guidance_scale=config.SD35_SCALES[args.target],
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1.,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                torch.cuda.synchronize()
                time_msg += (time.perf_counter() - t0)
                runs_msg += 1
                
                # --- Save Image ---
                image_name = f"msg.jpg"
                image_save_path = os.path.join(prompt_output_path, image_name)
                image.save(image_save_path)
                flush()

                # --- Log to JSON ---
                results_log[prompt_subdir_name] = positive_prompt

    finally:
        # --- Print Timing Results ---
        print("\n" + "="*50)
        print(f"AVERAGE GENERATION TIMES FOR {args.model.upper()}")
        print("="*50)
        if runs_cfg > 0:
            print(f"CFG: {time_cfg / runs_cfg:.2f} seconds/image (over {runs_cfg} images)")
        if runs_stg > 0:
            print(f"STG: {time_stg / runs_stg:.2f} seconds/image (over {runs_stg} images)")
        if runs_msg > 0:
            print(f"MSG: {time_msg / runs_msg:.2f} seconds/image (over {runs_msg} images)")
        print("="*50 + "\n")

        # --- Save JSON Log ---
        json_save_path = os.path.join(run_output_path, "results_log.json")
        print(f"\nSaving results log to: {json_save_path}")
        with open(json_save_path, "w") as f:
            json.dump(results_log, f, indent=4)

        print(f"\n--- Starting Automatic Evaluation for target: {args.target} ---")
        args.path = run_output_path
        args.detection_confidence = 0.7  # Default confidence for hand evaluation
        
        # Updated evaluation block to correctly handle all cases
        match args.target:
            case 'hands':
                evaluate_hands(args)
            case 'aesthetics':
                evaluate_aesthetics(args)
            case 'text':
                evaluate_text(args)
            case 'complex_text':
                evaluate_text(args)
            case _:
                print(f"No automatic evaluation script found for target: {args.target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', 'aesthetics'.")
    parser.add_argument("--model", type=str, required=True, help="Model type. 'flux', 'sd3', or 'pixart'.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output *directory* to save results.")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to a custom JSON dataset if target is not standard.")
    parser.add_argument("--max_prompts", type=int, default=100, help="Maximum number of prompts to process from the dataset.")

    args = parser.parse_args()
    main(args)