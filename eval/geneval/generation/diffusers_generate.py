"""Adapted from TODO"""

import argparse
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from eval.eval_hands import main as evaluate_hands
from eval.eval_aesthetics import main as evaluate_aesthetics
from eval.eval_text import main as evaluate_text
from src.SD3_custom_pipeline import StableDiffusion3Pipeline
from src.FLUX_custom_pipeline import FluxPipeline 
from src.PixartAlpha_custom_pipeline import PixArtAlphaPipeline
import config

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--skipped_layers",
        default=[],
        help="list of skipped layers for MSG",
    )
    parser.add_argument(
        "--layer_weights",
        default=[],
        help="list of layer weights for MSG",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model
    match opt.model:
        case 'flux':
            model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        case 'sd3':
            model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        case 'pixart':
            model = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        case 'sd35':
            model = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
        case _:
            raise ValueError(f"Unknown model: {opt.model}")
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.enable_attention_slicing()

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images

                match opt.model:
                    case 'flux':
                        model.multiskip = True
                        model.skipped_layers = config.FLUX_LAYERS[args.target]
                        model.layer_weights = config.FLUX_WEIGHTS[args.target]
                        image = model(
                            prompt=positive_prompt, 
                            negative_prompt=positive_prompt,    
                            true_cfg_scale=msg_guidance_scale,                 
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                        
                    case 'sd3':
                        model.multiskip = True
                        model.cfg_skip = False
                        model.skipped_layers = config.SD3_LAYERS[args.target]
                        model.layer_weights = config.SD3_WEIGHTS[args.target]
                        image = model(
                            prompt=positive_prompt, 
                            skip_guidance_layers=config.SD3_LAYERS[args.target],
                            skip_layer_guidance_scale=msg_guidance_scale,
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1., 
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]

                    case 'pixart':
                        model.multiskip = True
                        model.cfg_skip = False 
                        model.skipped_layers = config.PIXART_LAYERS[args.target]
                        model.layer_weights = config.PIXART_WEIGHTS[args.target]
                        image = model(
                            prompt=positive_prompt, 
                            skip_layer_guidance_scale = msg_guidance_scale,
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                    
                    case 'sd35':
                        model.multiskip = True
                        model.cfg_skip = False
                        model.skipped_layers = config.SD35_LAYERS[args.target]
                        model.layer_weights = config.SD35_WEIGHTS[args.target]
                        image = model(
                            prompt=positive_prompt, 
                            skip_guidance_layers=config.SD35_LAYERS[args.target],
                            skip_layer_guidance_scale=msg_guidance_scale,
                            skip_layer_guidance_start=0.,
                            skip_layer_guidance_stop=1., 
                            generator=torch.Generator("cuda").manual_seed(seed)
                        ).images[0]
                samples = model(
                    prompt,
                    height=opt.H,
                    width=opt.W,
                    num_inference_steps=opt.steps,
                    guidance_scale=opt.scale,
                    num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                    negative_prompt=opt.negative_prompt or None
                ).images
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
