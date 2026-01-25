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


def method_tag(method, cfg_scale, skip_scale, skip_layers, skip_weights):
    if method == "cfg":
        return f"cfg{fmt_float(cfg_scale)}"
    if method == "stg":
        return f"stg{fmt_float(skip_scale)}-l{skip_layers[0]}"
    if method == "naive":
        return f"naive{fmt_float(skip_scale)}-l{'-'.join(map(str, skip_layers))}"
    if method == "msg":
        weights_tag = ""
        if skip_weights is not None:
            weights_tag = "-w" + "-".join(fmt_float(w) for w in skip_weights)
        return f"msg{fmt_float(skip_scale)}-l{'-'.join(map(str, skip_layers))}{weights_tag}"
    return method


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


def main():
    parser = argparse.ArgumentParser(description="FID sweep for LightningDiT guidance variants.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--methods", type=str, default="cfg,stg,naive,msg")
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--skip-scale", type=float, default=None)
    parser.add_argument("--skip-layers", type=str, default=None)
    parser.add_argument("--skip-weights", type=str, default=None)
    parser.add_argument("--fid-num", type=int, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
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
    skip_scale = args.skip_scale if args.skip_scale is not None else base_config["sample"].get("skip_layer_guidance_scale", 1.0)
    skip_layers = parse_list(args.skip_layers, int)
    skip_weights = parse_list(args.skip_weights, float)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    alias = {"naive_msg": "naive", "naive-msg": "naive", "msg": "msg", "stg": "stg", "cfg": "cfg"}
    methods = [alias.get(m, m) for m in methods]

    accelerator = Accelerator()

    if "ckpt_path" not in base_config or base_config["ckpt_path"] in [None, ""]:
        raise ValueError("ckpt_path must be set in config or via --ckpt-path")
    if "fid_reference_file" not in base_config["data"]:
        raise ValueError("fid_reference_file must be set in config or via --fid-reference")

    if "downsample_ratio" in base_config["vae"]:
        latent_size = base_config["data"]["image_size"] // base_config["vae"]["downsample_ratio"]
    else:
        latent_size = base_config["data"]["image_size"] // 16

    model = build_model(base_config, latent_size)
    vae = VA_VAE(f'tokenizer/configs/{base_config["vae"]["model_name"]}.yaml')

    results = []
    base_exp_name = args.exp_name_prefix or base_config["train"]["exp_name"]

    for method in methods:
        run_config = deepcopy(base_config)
        sample_cfg = run_config["sample"]

        if args.fid_num is not None:
            sample_cfg["fid_num"] = args.fid_num
        if args.per_proc_batch_size is not None:
            sample_cfg["per_proc_batch_size"] = args.per_proc_batch_size

        if method == "cfg":
            sample_cfg["cfg_scale"] = cfg_scale
            sample_cfg["skip_layer_guidance_scale"] = 1.0
            sample_cfg["skip_guidance_layers"] = None
            sample_cfg["multiskip"] = False
            sample_cfg.pop("layer_weights", None)
            sample_cfg.pop("naive_skipping", None)
        elif method == "stg":
            if not skip_layers:
                raise ValueError("stg requires --skip-layers with at least one layer")
            if skip_scale == 1.0 and accelerator.process_index == 0:
                print_with_prefix("Warning: skip-scale=1.0 disables skip guidance.")
            sample_cfg["cfg_scale"] = 1.0
            sample_cfg["skip_layer_guidance_scale"] = skip_scale
            sample_cfg["skip_guidance_layers"] = [skip_layers[0]]
            sample_cfg["multiskip"] = False
            sample_cfg.pop("layer_weights", None)
            sample_cfg.pop("naive_skipping", None)
        elif method == "naive":
            if not skip_layers:
                raise ValueError("naive msg requires --skip-layers")
            if skip_scale == 1.0 and accelerator.process_index == 0:
                print_with_prefix("Warning: skip-scale=1.0 disables skip guidance.")
            sample_cfg["cfg_scale"] = 1.0
            sample_cfg["skip_layer_guidance_scale"] = skip_scale
            sample_cfg["skip_guidance_layers"] = skip_layers
            sample_cfg["multiskip"] = False
            sample_cfg["naive_skipping"] = True
            sample_cfg.pop("layer_weights", None)
        elif method == "msg":
            if not skip_layers:
                raise ValueError("msg requires --skip-layers")
            if skip_scale == 1.0 and accelerator.process_index == 0:
                print_with_prefix("Warning: skip-scale=1.0 disables skip guidance.")
            sample_cfg["cfg_scale"] = 1.0
            sample_cfg["skip_layer_guidance_scale"] = skip_scale
            sample_cfg["skip_guidance_layers"] = skip_layers
            sample_cfg["multiskip"] = True
            sample_cfg.pop("naive_skipping", None)
            if skip_weights is None:
                skip_weights = [1.0] * len(skip_layers)
            if len(skip_weights) != len(skip_layers):
                raise ValueError("skip_weights must match skip_layers length")
            sample_cfg["layer_weights"] = skip_weights
        else:
            raise ValueError(f"Unknown guidance method: {method}")

        tag = method_tag(method, sample_cfg["cfg_scale"], sample_cfg["skip_layer_guidance_scale"], skip_layers or [], skip_weights)
        run_config["train"]["exp_name"] = f"{base_exp_name}-{tag}"

        if accelerator.process_index == 0:
            print_with_prefix("Running", method, "exp_name=", run_config["train"]["exp_name"])

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
            results.append({"method": method, "fid": float(fid), "sample_dir": sample_dir})
            print_with_prefix("FID", method, "=", fid)

        accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        print_with_prefix("FID sweep results:")
        for row in results:
            print_with_prefix(row["method"], "fid=", row["fid"], "dir=", row["sample_dir"])
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
