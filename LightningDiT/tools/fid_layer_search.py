#!/usr/bin/env python
import argparse
import json
from copy import deepcopy
from typing import Optional

import torch
from accelerate import Accelerator

from inference import do_sample, load_config, print_with_prefix
from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE
from tools.calculate_fid import calculate_fid_given_paths


def parse_list(value: Optional[str], cast):
    if value is None:
        return None
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    return [cast(v) for v in items]


def fmt_float(value: float) -> str:
    return f"{value:.3g}".replace(".", "p")


def build_model(train_config, latent_size):
    return LightningDiT_models[train_config["model"]["model_type"]](
        input_size=latent_size,
        num_classes=train_config["data"]["num_classes"],
        use_qknorm=train_config["model"]["use_qknorm"],
        use_swiglu=train_config["model"].get("use_swiglu", False),
        use_rope=train_config["model"].get("use_rope", False),
        use_rmsnorm=train_config["model"].get("use_rmsnorm", False),
        wo_shift=train_config["model"].get("wo_shift", False),
        in_channels=train_config["model"].get("in_chans", 4),
        learn_sigma=train_config["model"].get("learn_sigma", False),
    )


def normalize_layers(layers, num_layers):
    if layers is None:
        return list(range(num_layers))
    bad_layers = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad_layers:
        raise ValueError(f"Layer indices out of range [0, {num_layers - 1}]: {bad_layers}")
    return layers


def main():
    parser = argparse.ArgumentParser(
        description="FID sweep for single-layer skip guidance over all DiT blocks."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--skip-scale", type=float, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--fid-num", type=int, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--exp-name-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--fid-reference", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.ckpt_path is not None:
        base_config["ckpt_path"] = args.ckpt_path
    if args.fid_reference is not None:
        base_config["data"]["fid_reference_file"] = args.fid_reference
    if args.output_dir is not None:
        base_config["train"]["output_dir"] = args.output_dir
    if args.seed is not None:
        base_config["train"]["global_seed"] = args.seed

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else base_config["sample"]["cfg_scale"]
    skip_scale = (
        args.skip_scale
        if args.skip_scale is not None
        else base_config["sample"].get("skip_layer_guidance_scale", 1.0)
    )
    layers = parse_list(args.layers, int)

    accelerator = Accelerator()

    if "ckpt_path" not in base_config or base_config["ckpt_path"] in [None, ""]:
        raise ValueError("ckpt_path must be set in config or via --ckpt-path")
    if "fid_reference_file" not in base_config["data"]:
        raise ValueError("fid_reference_file must be set in config or via --fid-reference")

    if skip_scale == 1.0 and accelerator.process_index == 0:
        print_with_prefix("Warning: skip-scale=1.0 disables skip guidance.")

    if "downsample_ratio" in base_config["vae"]:
        latent_size = base_config["data"]["image_size"] // base_config["vae"]["downsample_ratio"]
    else:
        latent_size = base_config["data"]["image_size"] // 16

    model = build_model(base_config, latent_size)
    vae = VA_VAE(f'tokenizer/configs/{base_config["vae"]["model_name"]}.yaml')

    num_layers = len(model.blocks)
    layers = normalize_layers(layers, num_layers)

    results = []
    base_exp_name = args.exp_name_prefix or base_config["train"]["exp_name"]
    scale_tag = fmt_float(skip_scale)

    for layer in layers:
        run_config = deepcopy(base_config)
        sample_cfg = run_config["sample"]

        if args.fid_num is not None:
            sample_cfg["fid_num"] = args.fid_num
        if args.per_proc_batch_size is not None:
            sample_cfg["per_proc_batch_size"] = args.per_proc_batch_size
        if args.num_sampling_steps is not None:
            sample_cfg["num_sampling_steps"] = args.num_sampling_steps

        sample_cfg["cfg_scale"] = cfg_scale
        sample_cfg["skip_layer_guidance_scale"] = skip_scale
        sample_cfg["skip_guidance_layers"] = [layer]
        sample_cfg["multiskip"] = False
        sample_cfg.pop("layer_weights", None)
        sample_cfg.pop("naive_skipping", None)

        run_config["train"]["exp_name"] = f"{base_exp_name}-layer{layer}-slg{scale_tag}"

        if accelerator.process_index == 0:
            print_with_prefix("Running layer", layer, "exp_name=", run_config["train"]["exp_name"])

        sample_dir = do_sample(
            run_config,
            accelerator,
            ckpt_path=run_config["ckpt_path"],
            model=model,
            vae=vae,
        )

        if accelerator.process_index == 0:
            fid_reference_file = run_config["data"]["fid_reference_file"]
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_dir],
                batch_size=50,
                dims=2048,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_workers=8,
                sp_len=run_config["sample"]["fid_num"],
            )
            results.append({"layer": layer, "fid": float(fid), "sample_dir": sample_dir})
            print_with_prefix("FID layer", layer, "=", fid)

        accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        print_with_prefix("Layer search results:")
        for row in results:
            print_with_prefix("layer", row["layer"], "fid=", row["fid"], "dir=", row["sample_dir"])
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
