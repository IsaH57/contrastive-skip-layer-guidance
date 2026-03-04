"""
Sampling Scripts of LightningDiT.

by Maple (Jingfeng Yao) from HUST-VL
"""

import os, math, json, pickle, logging, argparse, yaml, torch, numpy as np
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision
# local imports
from tokenizer.vavae import VA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from datasets.img_latent_dataset import ImgLatentDataset

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """
    Run sampling.
    """

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 9.0

    sample_cfg = train_config.get("sample", {})
    skip_guidance_layers = sample_cfg.get("skip_guidance_layers", None)
    if isinstance(skip_guidance_layers, str):
        skip_guidance_layers = [int(s) for s in skip_guidance_layers.split(",") if s.strip()]
    layer_weights = sample_cfg.get("layer_weights", None)
    if isinstance(layer_weights, str):
        layer_weights = [float(s) for s in layer_weights.split(",") if s.strip()]
    multiskip = bool(sample_cfg.get("multiskip", False))
    naive_skipping = sample_cfg.get("naive_skipping", None)
    if naive_skipping is not None:
        multiskip = not bool(naive_skipping)
    skip_layer_guidance_scale = float(sample_cfg.get("skip_layer_guidance_scale", 1.0))
    slg_active = skip_guidance_layers is not None and skip_layer_guidance_scale != 1.0
    if slg_active:
        folder_name += f"-slg{skip_layer_guidance_scale:.2f}"

    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)
        if slg_active:
            print_with_prefix(
                "Skip-layer guidance:",
                f"layers={skip_guidance_layers}",
                f"scale={skip_layer_guidance_scale}",
                f"mode={'multiskip' if multiskip else 'naive'}",
                f"weights={layer_weights}",
            )

    if accelerator.process_index == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Figure out how many samples we need and where to resume.
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    done_iterations = int(num_samples // global_batch_size)
    resumed_num_samples = done_iterations * global_batch_size
    remaining_samples = max(total_samples - resumed_num_samples, 0)

    if accelerator.process_index == 0:
        print_with_prefix(
            f"Sampling plan: target={total_samples} existing={num_samples} "
            f"resume_from={resumed_num_samples} remaining={remaining_samples}"
        )

    if num_samples >= total_samples:
        if accelerator.process_index == 0:
            print_with_prefix(f"Found {num_samples} PNG files in {sample_folder_dir}, skip sampling.")
        return sample_folder_dir

    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(done_iterations, iterations)
    if not demo_sample_mode and rank == 0:
        pbar = tqdm(pbar, total=iterations, initial=done_iterations)

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    if vae is None:
        vae = VA_VAE(
            f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml',
        )
        if accelerator.process_index == 0:
            print_with_prefix('Loaded VAE model')

    using_cfg = cfg_scale > 1.0 or slg_active
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)


    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for label in tqdm([975, 3, 207, 387, 388, 88, 979, 279], desc="Generating Demo Samples"):
                z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
                y = torch.tensor([label], device=device)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * 1, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(
                    y=y,
                    cfg_scale=cfg_scale,
                    cfg_interval=False,
                    cfg_interval_start=cfg_interval_start,
                    skip_guidance_layers=skip_guidance_layers,
                    skip_layer_guidance_scale=skip_layer_guidance_scale,
                    multiskip=multiskip,
                    layer_weights=layer_weights,
                )
                model_fn = model.forward_with_cfg
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)
                images.append(samples)
            # Combine 8 images into a 2x4 grid
            os.makedirs('demo_images', exist_ok=True)
            # Stack all images into a large numpy array
            all_images = np.stack([img[0] for img in images])  # Take first image from each batch            
            # Rearrange into 2x4 grid
            h, w = all_images.shape[1:3]
            grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 4)  # Calculate position in 2x4 grid
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
                
            # Save the combined image
            Image.fromarray(grid).save('demo_images/demo_samples.png')

            return None
    else:
        for step in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)
            
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(
                    y=y,
                    cfg_scale=cfg_scale,
                    cfg_interval=True,
                    cfg_interval_start=cfg_interval_start,
                    skip_guidance_layers=skip_guidance_layers,
                    skip_layer_guidance_scale=skip_layer_guidance_scale,
                    multiskip=multiskip,
                    layer_weights=layer_weights,
                )
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * accelerator.num_processes + accelerator.process_index + step * global_batch_size
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            accelerator.wait_for_everyone()

    return sample_folder_dir

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16

    # get model
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
    )

    # naive sample
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
    
    if not args.demo:
        # calculate FID
        # Important: FID is only for reference, please use ADM evaluation for paper reporting
        if accelerator.process_index == 0:
            from tools.calculate_fid import calculate_fid_given_paths
            print_with_prefix('Calculating FID with {} number of samples'.format(train_config['sample']['fid_num']))
            assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
            fid_reference_file = train_config['data']['fid_reference_file']
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=50,
                dims=2048,
                device='cuda',
                num_workers=8,
                sp_len = train_config['sample']['fid_num']
            )
            print_with_prefix('fid=',fid)
