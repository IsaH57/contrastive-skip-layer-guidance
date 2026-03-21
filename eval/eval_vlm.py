"""VLM-based evaluator for tasks that need semantic/style judgments.

This script mirrors the existing eval scripts:
- Reads an experiment directory structured as prompt folders with image files.
- Scores each image mode (cfg, layer_0, layer_1, ...).
- Saves per-prompt scores to CSV.
- Prints per-mode averages at the end.
"""

import argparse
import gc
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from eval.system_prompts import (
        BACKGROUND as BACKGROUND_SYSTEM_PROMPT,
        SYMMETRY as SYMMETRY_SYSTEM_PROMPT,
        UKIYO as UKIYO_SYSTEM_PROMPT,
    )
except Exception:
    UKIYO_SYSTEM_PROMPT = ""
    BACKGROUND_SYSTEM_PROMPT = ""
    SYMMETRY_SYSTEM_PROMPT = ""

SUPPORTED_TASKS = ("ukiyo", "background", "symmetry")
DEFAULT_LOCAL_INTERNVL_CACHE = (
    "/export/scratch/ru63zus/hub/internVL-14B/models--OpenGVLab--InternVL3-14B"
)

DEFAULT_TASK_SYSTEM_PROMPTS: Dict[str, str] = {
    "ukiyo": UKIYO_SYSTEM_PROMPT,
    "background": BACKGROUND_SYSTEM_PROMPT,
    "symmetry": SYMMETRY_SYSTEM_PROMPT,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_transform(input_size: int):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios,
    width: int,
    height: int,
    image_size: int,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def mode_sort_key(mode: str):
    if mode == "cfg":
        return (0, 0)
    layer_match = re.fullmatch(r"layer_(\d+)", mode)
    if layer_match:
        return (1, int(layer_match.group(1)))
    return (2, mode)


def discover_prompt_dirs_and_modes(experiment_root: str) -> Tuple[List[str], List[str]]:
    prompt_dirs = sorted(
        [
            d
            for d in os.listdir(experiment_root)
            if os.path.isdir(os.path.join(experiment_root, d)) and not d.startswith(".")
        ]
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt directories found in: {experiment_root}")

    image_modes: List[str] = []
    sampled_prompt_path = None
    for prompt_dir in prompt_dirs:
        prompt_path = os.path.join(experiment_root, prompt_dir)
        sampled_modes = []
        for entry in os.listdir(prompt_path):
            file_path = os.path.join(prompt_path, entry)
            stem, ext = os.path.splitext(entry)
            if os.path.isfile(file_path) and ext.lower() in IMAGE_EXTENSIONS:
                sampled_modes.append(stem)
        if sampled_modes:
            sampled_prompt_path = prompt_path
            image_modes = sorted(set(sampled_modes), key=mode_sort_key)
            break

    if not image_modes:
        raise FileNotFoundError(
            "No image files found in prompt directories under "
            f"{experiment_root}. Last checked path: {sampled_prompt_path}"
        )
    return prompt_dirs, image_modes


def resolve_image_path(prompt_path: str, mode: str) -> Optional[str]:
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(prompt_path, f"{mode}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


def resolve_vlm_model_path(model_path: str) -> str:
    """Resolve HF cache-style local model paths to a concrete snapshot folder."""
    if not os.path.isdir(model_path):
        return model_path

    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    snapshots_root = os.path.join(model_path, "snapshots")
    if not os.path.isdir(snapshots_root):
        return model_path

    snapshot_dirs = sorted(
        [
            os.path.join(snapshots_root, d)
            for d in os.listdir(snapshots_root)
            if os.path.isdir(os.path.join(snapshots_root, d))
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_dirs:
        if os.path.isfile(os.path.join(snapshot_dir, "config.json")):
            print(f"Resolved VLM cache path to snapshot: {snapshot_dir}")
            return snapshot_dir

    raise FileNotFoundError(
        f"No valid snapshot with config.json found under: {snapshots_root}"
    )


def build_scoring_prompt(task: str, generation_prompt: str, system_prompt: str) -> str:
    if system_prompt.strip():
        system_block = f"{system_prompt.strip()}\n\n"
    else:
        system_block = ""

    return (
        "<image>\n"
        f"{system_block}"
        f"You are judging one generated image for the task '{task}'.\n"
        f'Generation prompt: "{generation_prompt}"\n\n'
        "Score how well this image satisfies the task on a strict 0-100 scale.\n"
        "0 = complete failure, 100 = perfect success.\n"
        "Return only one final line exactly like this:\n"
        "SCORE: <number>/100"
    )


def normalize_score(raw_value: float, denominator: Optional[float] = None) -> float:
    score = raw_value
    if denominator is not None:
        score = (raw_value / denominator) * 100.0
    elif score <= 10.0:
        # If denominator is omitted and the model returns 0-10, normalize to 0-100.
        score = score * 10.0

    return max(0.0, min(100.0, score))


def parse_score(response_text: str) -> Optional[float]:
    if not response_text:
        return None

    # 1) Preferred format: SCORE: X/100 (or /10)
    labeled_with_denom = re.search(
        r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*/\s*(100|10)",
        response_text,
        flags=re.IGNORECASE,
    )
    if labeled_with_denom:
        value = float(labeled_with_denom.group(1))
        denom = float(labeled_with_denom.group(2))
        return normalize_score(value, denom)

    # 2) Labeled score without denominator: SCORE: X
    labeled_no_denom = re.search(
        r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        response_text,
        flags=re.IGNORECASE,
    )
    if labeled_no_denom:
        value = float(labeled_no_denom.group(1))
        return normalize_score(value)

    # 3) JSON-ish fallback: {"score": X}
    json_like = re.search(
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        response_text,
        flags=re.IGNORECASE,
    )
    if json_like:
        value = float(json_like.group(1))
        return normalize_score(value)

    # 4) Last-line ratio fallback: X/100 or X/10
    last_line = response_text.strip().splitlines()[-1] if response_text.strip() else ""
    ratio_last_line = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(100|10)", last_line)
    if ratio_last_line:
        value = float(ratio_last_line.group(1))
        denom = float(ratio_last_line.group(2))
        return normalize_score(value, denom)

    return None


class InternVLEvaluator:
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        input_size: int,
        max_tiles: int,
        max_new_tokens: int,
        use_flash_attn: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ):
        self.model_path = resolve_vlm_model_path(model_path)
        self.dtype = dtype
        self.input_size = input_size
        self.max_tiles = max_tiles
        self.max_new_tokens = max_new_tokens

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for VLM evaluation.")

        print(f"Loading VLM model: {self.model_path}")
        local_only = os.path.isdir(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_only,
        )

        model_kwargs = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if use_flash_attn:
            model_kwargs["use_flash_attn"] = True

        try:
            self.model = (
                AutoModel.from_pretrained(
                    self.model_path, local_files_only=local_only, **model_kwargs
                )
                .eval()
                .cuda()
            )
        except TypeError:
            if "use_flash_attn" in model_kwargs:
                print(
                    "[Warning] 'use_flash_attn' is not supported by this model/version. "
                    "Retrying without it."
                )
                model_kwargs.pop("use_flash_attn")
                self.model = (
                    AutoModel.from_pretrained(
                        self.model_path, local_files_only=local_only, **model_kwargs
                    )
                    .eval()
                    .cuda()
                )
            else:
                raise

        self.generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            self.generation_config.update(temperature=temperature, top_p=top_p)

    def score_image(
        self,
        image_path: str,
        task: str,
        generation_prompt: str,
        system_prompt: str,
    ) -> Tuple[Optional[float], str]:
        question = build_scoring_prompt(task, generation_prompt, system_prompt)
        pixel_values = load_image(
            image_file=image_path, input_size=self.input_size, max_num=self.max_tiles
        ).to(self.dtype).cuda()

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
                history=None,
                return_history=False,
            )

        if isinstance(response, tuple):
            response_text = response[0]
        else:
            response_text = response

        score = parse_score(response_text)
        return score, response_text


def main(args):
    experiment_root = args.path
    task = args.task
    model_name = args.model
    output_csv = f"{model_name}_{task}_vlm_ratings.csv"

    print(f"Experiment path: {experiment_root}")
    print(f"Task: {task}")
    print(f"Generator model label: {model_name}")
    if args.num_evals_per_image > 1 and not args.do_sample:
        print(
            "[Info] num_evals_per_image > 1 requested with deterministic decoding. "
            "Enabling sampling to avoid identical repeated scores."
        )
        args.do_sample = True
    print(f"Evaluations per image: {args.num_evals_per_image}")
    print(
        "VLM decoding config: "
        f"do_sample={args.do_sample}, temperature={args.temperature}, top_p={args.top_p}"
    )

    prompt_dirs, image_modes = discover_prompt_dirs_and_modes(experiment_root)
    print(f"Found {len(prompt_dirs)} prompt directories.")
    print(f"Image modes: {image_modes}")

    prompt_system_map = {
        "ukiyo": args.ukiyo_system_prompt,
        "background": args.background_system_prompt,
        "symmetry": args.symmetry_system_prompt,
    }
    system_prompt = prompt_system_map[task]
    if not system_prompt.strip():
        print(
            f"[Warning] System prompt for task '{task}' is blank. "
            "Please fill it in for best results."
        )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    evaluator = InternVLEvaluator(
        model_path=args.vlm_model_path,
        dtype=dtype,
        input_size=args.input_size,
        max_tiles=args.max_tiles,
        max_new_tokens=args.max_new_tokens,
        use_flash_attn=args.use_flash_attn,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results_log_path = os.path.join(experiment_root, "results_log.json")
    prompt_lookup = {}
    if os.path.isfile(results_log_path):
        with open(results_log_path, "r", encoding="utf-8") as f:
            prompt_lookup = json.load(f)
    else:
        print(
            f"[Warning] Missing results_log.json at {results_log_path}. "
            "Using prompt directory names as prompt text."
        )

    print("Starting VLM evaluation...")
    rows = []
    for prompt_dir in tqdm(prompt_dirs):
        prompt_path = os.path.join(experiment_root, prompt_dir)
        prompt_text = prompt_lookup.get(prompt_dir, prompt_dir)
        row = {"prompt": prompt_text}

        for mode in image_modes:
            image_path = resolve_image_path(prompt_path, mode)
            if image_path is None:
                row[mode] = None
                print(f"[Warning] Image not found for mode '{mode}' in {prompt_path}")
                continue

            mode_scores: List[Optional[float]] = []
            for eval_idx in range(args.num_evals_per_image):
                score = None
                last_response = ""
                for attempt_idx in range(args.max_retries):
                    try:
                        score, response_text = evaluator.score_image(
                            image_path=image_path,
                            task=task,
                            generation_prompt=prompt_text,
                            system_prompt=system_prompt,
                        )
                        last_response = response_text
                        if score is not None:
                            break
                        print(
                            f"[Warning] Could not parse score ({prompt_dir}/{mode}, "
                            f"eval {eval_idx + 1}/{args.num_evals_per_image}) "
                            f"attempt {attempt_idx + 1}/{args.max_retries}"
                        )
                    except Exception as exc:
                        print(
                            f"[Error] Failed to score {image_path} "
                            f"(eval {eval_idx + 1}/{args.num_evals_per_image}, "
                            f"attempt {attempt_idx + 1}/{args.max_retries}): {exc}"
                        )
                        flush()

                if score is None and last_response:
                    tail = last_response.strip().replace("\n", " ")
                    print(
                        f"[Warning] Last unparsed response ({prompt_dir}/{mode}, "
                        f"eval {eval_idx + 1}/{args.num_evals_per_image}): {tail[:220]}"
                    )

                mode_scores.append(score)
                if args.num_evals_per_image > 1:
                    row[f"{mode}_eval_{eval_idx + 1}"] = score
                flush()

            valid_mode_scores = [s for s in mode_scores if s is not None]
            row[mode] = (
                float(statistics.mean(valid_mode_scores)) if valid_mode_scores else None
            )
            if args.num_evals_per_image > 1:
                row[f"{mode}_valid_evals"] = len(valid_mode_scores)
                row[f"{mode}_std"] = (
                    float(statistics.pstdev(valid_mode_scores)) if valid_mode_scores else None
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    averages = df[image_modes].mean(numeric_only=True)
    print("\n--- Overall Averages ---")
    print(averages)

    if not averages.empty:
        print(f"\nGlobal mean score across modes: {averages.mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate layer-ablation images with a VLM on a 0-100 scale."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help="Task to evaluate with task-specific system prompt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Generator model name label (e.g., flux, sd3, sd35) used for output naming.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root directory path for generated images (layer ablation output).",
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default=DEFAULT_LOCAL_INTERNVL_CACHE,
        help=(
            "Local VLM path or HF ID. If this points to an HF cache dir "
            "(models--.../snapshots), the latest valid snapshot is auto-resolved."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Compute dtype for VLM inference.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Enable flash attention if supported by the VLM.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=448,
        help="InternVL patch input resolution.",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=12,
        help="Maximum dynamic tiles used by InternVL preprocessing.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for each score response.",
    )
    parser.add_argument(
        "--num_evals_per_image",
        type=int,
        default=1,
        help="Number of repeated VLM evaluations per image. Final mode score is the mean.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable stochastic decoding for repeated evaluations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature used when --do_sample is enabled.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p used when --do_sample is enabled.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Retries per image if score parsing fails.",
    )

    parser.add_argument(
        "--ukiyo_system_prompt",
        type=str,
        default=DEFAULT_TASK_SYSTEM_PROMPTS["ukiyo"],
        help="System prompt for ukiyo evaluation.",
    )
    parser.add_argument(
        "--background_system_prompt",
        type=str,
        default=DEFAULT_TASK_SYSTEM_PROMPTS["background"],
        help="System prompt for background evaluation.",
    )
    parser.add_argument(
        "--symmetry_system_prompt",
        type=str,
        default=DEFAULT_TASK_SYSTEM_PROMPTS["symmetry"],
        help="System prompt for symmetry evaluation.",
    )

    parsed_args = parser.parse_args()
    if parsed_args.num_evals_per_image < 1:
        parser.error("--num_evals_per_image must be >= 1.")
    if parsed_args.temperature <= 0:
        parser.error("--temperature must be > 0.")
    if not (0 < parsed_args.top_p <= 1):
        parser.error("--top_p must be in (0, 1].")

    main(parsed_args)
