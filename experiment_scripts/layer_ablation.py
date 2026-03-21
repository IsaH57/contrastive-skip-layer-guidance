import json
import torch
import gc
import sys
import os
import random
import torch.nn.functional as F
import argparse
from datetime import datetime 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.SD3_custom_pipeline import StableDiffusion3Pipeline
from src.FLUX_custom_pipeline import FluxPipeline 
from src.PixartAlpha_custom_pipeline import PixArtAlphaPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

FID_CANDIDATE_LAYERS = {'flux': [12,14,15], 
                        'sd3': [6,7,9],
                        'sd35': [2,6,9],
                        'pixart':[3,5,8]}

DEFAULT_FLUX_MODEL_PATH = "/export/scratch/ru63zus/hub/flux"

def flush():
    """Helper function to free up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def _is_nonempty_file(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def _is_standard_prompt_complete(prompt_dir, expected_layer_files, require_cfg):
    if not os.path.isdir(prompt_dir):
        return False
    if require_cfg and not _is_nonempty_file(os.path.join(prompt_dir, "cfg.jpg")):
        return False
    for layer_file in expected_layer_files:
        if not _is_nonempty_file(os.path.join(prompt_dir, layer_file)):
            return False
    return True

def _is_geneval_leaf_complete(root, idx):
    leaf = os.path.join(root, f"{idx:05d}")
    return (
        os.path.isdir(leaf)
        and _is_nonempty_file(os.path.join(leaf, "metadata.jsonl"))
        and _is_nonempty_file(os.path.join(leaf, "grid.png"))
        and _is_nonempty_file(os.path.join(leaf, "samples", "0000.png"))
    )

def _is_geneval_prompt_complete(run_output_path, idx, layer_indices):
    if not _is_geneval_leaf_complete(os.path.join(run_output_path, "cfg"), idx):
        return False
    for layer in layer_indices:
        if not _is_geneval_leaf_complete(os.path.join(run_output_path, f"layer_{layer}"), idx):
            return False
    return True

def _resolve_flux_model_path(flux_model_path):
    if not flux_model_path:
        raise ValueError("FLUX model path is empty. Pass --flux_model_path.")
    if not os.path.isdir(flux_model_path):
        raise ValueError(f"FLUX model path does not exist: {flux_model_path}")
    model_index = os.path.join(flux_model_path, "model_index.json")
    if not os.path.isfile(model_index):
        raise ValueError(
            f"FLUX model directory is missing model_index.json: {flux_model_path}"
        )
    return flux_model_path

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

def _persist_results_log(json_save_path, results_log):
    """
    Atomically persist the results log so it survives interruptions.
    """
    tmp_path = f"{json_save_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results_log, f, indent=4)
    os.replace(tmp_path, json_save_path)

def main(args):
    print(f'PERFORMING LAYER ABLATION FOR TARGET: {args.target}')
    print(f'USING MODEL: {args.model}')
    print(f'MAX PROMPTS: {args.max_prompts}')
    print(f'USING CFG SKIP: {args.cfg_skip}')

    # Load Dataset
    match args.target: 
        case 'ukiyo':
            dataset = json.load(open('prompt_datasets/VLM_targets/ukiyo-e/ukyio_pairs.json', "r"))
        case 'background':
            dataset = json.load(open('prompt_datasets/VLM_targets/background_separation/background_pairs.json', "r"))
        case 'symmetry':
            dataset = json.load(open('prompt_datasets/VLM_targets/symmetry_regularity/regularity_pairs.json', "r"))
        case 'coco':
            dataset = json.load(open('/export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/prompt_datasets/coco/captions_val2014_prompts.json'))
        case 'text': 
            dataset = json.load(open('prompt_datasets/text_pairs.json', "r"))
        case 'complex_text':
            dataset = json.load(open('prompt_datasets/text/complex_prompt_pairs.json', "r"))
        case 'hands':
            dataset = json.load(open('prompt_datasets/hands_pairs.json', "r"))
        case 'aesthetics':
            dataset = json.load(open('prompt_datasets/aesthetics_pairs.json', "r"))
        case 'binding':
            with open('prompt_datasets/geneval/attribute_binding_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'counting':
            with open('prompt_datasets/geneval/counting_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case 'position':
            with open('prompt_datasets/geneval/position_metadata.jsonl') as fp:
                dataset = [json.loads(line) for line in fp]
        case _: 
            if args.dataset_path != '':
                dataset = json.load(open(args.dataset_path, "r"))
            else:
                print("No valid target or dataset_path provided.")
                raise ValueError("Invalid dataset target or path")
            
    if args.target == 'coco':
        rng = random.Random(42)  # fixed seed -> same shuffle every time
        rng.shuffle(dataset)
    else:
        random.shuffle(dataset)

    # Process dataset slice
    prompts_to_process = dataset[:args.max_prompts]
    num_prompts = len(prompts_to_process)

    # --- Resolve Output Directory ---
    if args.resume_output_path:
        run_output_path = args.resume_output_path
        if not os.path.isdir(run_output_path):
            raise ValueError(f"--resume_output_path does not exist or is not a directory: {run_output_path}")
        print(f"Resuming into existing output directory: {run_output_path}")
    else:
        if not args.output_path:
            raise ValueError("Either --output_path (new run) or --resume_output_path (resume run) must be provided.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"{args.model}_{args.target}_cfgskip_{args.cfg_skip}_layer_ablation_{timestamp}"
        run_output_path = os.path.join(args.output_path, run_dir_name)
        os.makedirs(run_output_path, exist_ok=True)
    print(f"Saving results to: {run_output_path}")

    # --- JSON Log ---
    results_log = {}
    json_save_path = os.path.join(run_output_path, "results_log.json")
    if os.path.isfile(json_save_path):
        try:
            with open(json_save_path, "r") as f:
                results_log = json.load(f)
            print(f"Loaded existing results log with {len(results_log)} entries.")
        except json.JSONDecodeError:
            print("Existing results_log.json is invalid JSON; starting with an empty log.")
            results_log = {}
    _persist_results_log(json_save_path, results_log)

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
    layer_indices = (
        list(range(len(pipe.transformer.transformer_blocks)))
        if args.target != "coco"
        else FID_CANDIDATE_LAYERS[args.model]
    )
    expected_layer_files = [f"layer_{layer}.jpg" for layer in layer_indices]
    skipped_complete = 0
    generated_prompts = 0

    try:
        for idx, data in enumerate(prompts_to_process):
            positive_prompt = data['positive'] if not is_geneval else data['prompt']
            print(f"\n--- Processing Prompt {idx+1}/{num_prompts} ---")
            print(f"Prompt: {positive_prompt}")

            prompt_subdir_name = f"{idx:05d}"
            prompt_output_path = os.path.join(run_output_path, prompt_subdir_name)
            prompt_complete = (
                _is_geneval_prompt_complete(run_output_path, idx, layer_indices)
                if is_geneval
                else _is_standard_prompt_complete(
                    prompt_output_path,
                    expected_layer_files=expected_layer_files,
                    require_cfg=(args.target != "coco"),
                )
            )
            if prompt_complete:
                print("Prompt already complete in output folder; skipping.")
                skipped_complete += 1
                if is_geneval:
                    results_log[f"cfg/{prompt_subdir_name}"] = positive_prompt
                    for layer in layer_indices:
                        results_log[f"layer_{layer}/{prompt_subdir_name}"] = positive_prompt
                else:
                    results_log[prompt_subdir_name] = positive_prompt
                _persist_results_log(json_save_path, results_log)
                continue

            # Use one seed per prompt across all configs for consistency
            seed = random.randint(0, 2**32 - 1)
            generated_prompts += 1

            if is_geneval:
                # --- Geneval branch: separate subfolders per config (cfg + each layer) ---
                # 1) CFG (no skips)
                match args.model:
                    case 'flux':
                        pipe.skipped_layers = []
                        img = pipe(prompt=positive_prompt,
                                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                    case 'sd3':
                        pipe.multiskip = False
                        pipe.cfg_skip = False
                        img = pipe(prompt=positive_prompt,
                                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                    case 'pixart':
                        pipe.multiskip = False
                        pipe.cfg_skip = False
                        pipe.skipped_layers = []
                        img = pipe(prompt=positive_prompt,
                                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                    case 'sd35':
                        pipe.multiskip = False
                        pipe.cfg_skip = False
                        img = pipe(prompt=positive_prompt,
                                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]

                cfg_root = os.path.join(run_output_path, "cfg")
                os.makedirs(cfg_root, exist_ok=True)
                leaf = _save_geneval_leaf(cfg_root, idx, data, img)
                results_log[f"cfg/{os.path.basename(leaf)}"] = positive_prompt
                flush()

                # 2) Per-layer folders
                num_layers = len(pipe.transformer.transformer_blocks)
                for layer in range(num_layers):
                    print(f"Generating for skipped layer: {layer}")

                    match args.model:
                        case 'flux':
                            pipe.skipped_layers = [layer]
                            img = pipe(prompt=positive_prompt,
                                       negative_prompt=positive_prompt,
                                       true_cfg_scale=2.0,
                                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                        case 'sd3':
                            pipe.multiskip = True
                            pipe.cfg_skip = False
                            pipe.layer_weights = [1.0]
                            img = pipe(prompt=positive_prompt,
                                       skip_guidance_layers=[layer],
                                       skip_layer_guidance_scale=2.0,
                                       skip_layer_guidance_start=0.,
                                       skip_layer_guidance_stop=1.,
                                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                        case 'pixart':
                            pipe.multiskip = True
                            pipe.cfg_skip = False
                            pipe.layer_weights = [1.0]
                            pipe.skipped_layers = [layer]
                            img = pipe(prompt=positive_prompt,
                                       skip_layer_guidance_scale=2.0,
                                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                        case 'sd35':
                            pipe.multiskip = True
                            pipe.cfg_skip = False
                            pipe.layer_weights = [1.0]
                            img = pipe(prompt=positive_prompt,
                                       skip_guidance_layers=[layer],
                                       skip_layer_guidance_scale=2.0,
                                       skip_layer_guidance_start=0.,
                                       skip_layer_guidance_stop=1.,
                                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]

                    layer_root = os.path.join(run_output_path, f"layer_{layer}")
                    os.makedirs(layer_root, exist_ok=True)
                    leaf = _save_geneval_leaf(layer_root, idx, data, img)
                    results_log[f"layer_{layer}/{os.path.basename(leaf)}"] = positive_prompt
                    flush()

            else:
                if args.target == 'coco':
                    # --- Original branch unchanged for non-Geneval targets ---
                    os.makedirs(prompt_output_path, exist_ok=True)
                    for layer in FID_CANDIDATE_LAYERS[args.model]:
                        print(f"Generating for skipped layer: {layer}")
            
                        match args.model:
                            case 'flux':
                                pipe.skipped_layers = [layer]   
                                image = pipe(
                                    prompt=positive_prompt, 
                                    negative_prompt=positive_prompt,
                                    true_cfg_scale=2.0,
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'sd3':
                                pipe.multiskip = True
                                pipe.cfg_skip = False
                                pipe.layer_weights = [1.0]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_guidance_layers=[layer],
                                    skip_layer_guidance_scale=2.0,
                                    skip_layer_guidance_start=0.,
                                    skip_layer_guidance_stop=1., 
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'pixart':
                                pipe.multiskip = True
                                pipe.cfg_skip = False 
                                pipe.layer_weights = [1.0]
                                pipe.skipped_layers = [layer]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_layer_guidance_scale = 2.0,
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'sd35':
                                pipe.multiskip = True
                                pipe.cfg_skip = False
                                pipe.layer_weights = [1.0]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_guidance_layers=[layer],
                                    skip_layer_guidance_scale=2.0,
                                    skip_layer_guidance_start=0.,
                                    skip_layer_guidance_stop=1., 
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                        
                        image_name = f"layer_{layer}.jpg"
                        image_save_path = os.path.join(prompt_output_path, image_name)
                        image.save(image_save_path)
                        flush()

                else:
                    # --- Original branch unchanged for non-Geneval targets ---
                    os.makedirs(prompt_output_path, exist_ok=True)

                    # Generate CFG image without MSG
                    match args.model:
                        case 'flux':
                            pipe.skipped_layers = []   
                            image = pipe(
                                prompt=positive_prompt, 
                                generator=torch.Generator("cuda").manual_seed(seed)
                            ).images[0]
                        case 'sd3':
                            pipe.multiskip = False
                            pipe.cfg_skip = False
                            image = pipe(
                                prompt=positive_prompt, 
                                generator=torch.Generator("cuda").manual_seed(seed)
                            ).images[0]
                        case 'pixart':
                            pipe.multiskip = False
                            pipe.cfg_skip = False
                            pipe.skipped_layers = []
                            image = pipe(
                                prompt=positive_prompt, 
                                generator=torch.Generator("cuda").manual_seed(seed)
                            ).images[0]
                        case 'sd35':
                            pipe.multiskip = False
                            pipe.cfg_skip = False
                            image = pipe(
                                prompt=positive_prompt, 
                                generator=torch.Generator("cuda").manual_seed(seed)
                            ).images[0]

                    image_name = f"cfg.jpg"
                    image_save_path = os.path.join(prompt_output_path, image_name)
                    image.save(image_save_path)
                    flush()
                    
                    for layer in range(len(pipe.transformer.transformer_blocks)):
                        print(f"Generating for skipped layer: {layer}")
            
                        match args.model:
                            case 'flux':
                                pipe.skipped_layers = [layer]   
                                image = pipe(
                                    prompt=positive_prompt, 
                                    negative_prompt=positive_prompt,
                                    true_cfg_scale=2.5,
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'sd3':
                                pipe.multiskip = True
                                pipe.cfg_skip = False
                                pipe.layer_weights = [1.0]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_guidance_layers=[layer],
                                    skip_layer_guidance_scale=2.5,
                                    skip_layer_guidance_start=0.,
                                    skip_layer_guidance_stop=1., 
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'pixart':
                                pipe.multiskip = True
                                pipe.cfg_skip = False 
                                pipe.layer_weights = [1.0]
                                pipe.skipped_layers = [layer]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_layer_guidance_scale = 2.5,
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                            case 'sd35':
                                pipe.multiskip = True
                                pipe.cfg_skip = False
                                pipe.layer_weights = [1.0]
                                image = pipe(
                                    prompt=positive_prompt, 
                                    skip_guidance_layers=[layer],
                                    skip_layer_guidance_scale=2.5,
                                    skip_layer_guidance_start=0.,
                                    skip_layer_guidance_stop=1., 
                                    generator=torch.Generator("cuda").manual_seed(seed)
                                ).images[0]
                        
                        image_name = f"layer_{layer}.jpg"
                        image_save_path = os.path.join(prompt_output_path, image_name)
                        image.save(image_save_path)
                        flush()

                results_log[prompt_subdir_name] = positive_prompt
                _persist_results_log(json_save_path, results_log)

            if is_geneval:
                _persist_results_log(json_save_path, results_log)

    finally:
        print(f"\nSaving results log to: {json_save_path}")
        _persist_results_log(json_save_path, results_log)

        print(f"Prompts generated this run: {generated_prompts}")
        print(f"Prompts skipped as complete: {skipped_complete}")
        print(f"--- Run complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', 'aesthetics'.")
    parser.add_argument("--model", type=str, required=True, help="Model type. 'flux', 'sd3', or 'pixart'.")
    parser.add_argument("--output_path", type=str, default="", help="Path to output *parent* directory for a new run.")
    parser.add_argument("--resume_output_path", type=str, default="", help="Path to an existing run directory to resume.")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to a custom JSON dataset if target is not standard.")
    parser.add_argument("--max_prompts", type=int, default=100, help="Maximum number of prompts (from dataset start) to ensure are generated.")
    parser.add_argument("--cfg_skip", type=bool, default=False, help="Whether to use CFG skip behavior in SD3 and Pixart models.")
    parser.add_argument(
        "--flux_model_path",
        type=str,
        default=DEFAULT_FLUX_MODEL_PATH,
        help="Local path to a FLUX diffusers model directory (used when --model flux).",
    )

    args = parser.parse_args()
    main(args)
