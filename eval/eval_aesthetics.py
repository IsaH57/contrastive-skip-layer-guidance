""" 
This code evaluates the aesthetic quality of generated images using the Simple Aesthetics Predictor based on CLIP.

It processes images from an experiment folder structured by prompts and seeds, computes aesthetic scores per image at different CFG and SLG guidance scales,
then saves detailed CSV results and aggregate statistics similar to the text visibility and hand quality eval pipelines.
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import itertools
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1

multilayer_ablation = True

# --- AestheticsValidator class ---
class AestheticsValidator:
    def __init__(self, model_id: str = "shunk031/aesthetics-predictor-v1-vit-large-patch14"):
        """
        Initialize Simple Aesthetics Predictor pipeline
        
        Args:
            model_id: HuggingFace model identifier for the aesthetics predictor
        """
        print(f"Loading aesthetics predictor: {model_id}")
        self.predictor = AestheticsPredictorV1.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = self.predictor.to(self.device)
        print(f"Using device: {self.device}")

    def evaluate_aesthetics(self, image_path: str) -> float:
        """
        Evaluate aesthetic quality of an image.

        Args:
            image_path: Path to input image

        Returns:
            Aesthetic score (float)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference for the image
            with torch.no_grad():
                outputs = self.predictor(**inputs)
            
            # Return the aesthetic score as a float
            return float(outputs.logits.item())
            
        except Exception as e:
            print(f"[Error] {image_path}: {e}")
            return None

# --- Config ---
EXPERIMENT_ROOT = '/export/home/ru63zus/repos/contrastive-skip-layer-guidance/experiments/flux_multiskip_results_20251005_080303'
MODEL = 'FLUX'
OUTPUT_CSV = f'{MODEL}_aesthetic_quality_ablations.csv'
IMAGE_TYPES = [filename[:-4] for filename in os.listdir(EXPERIMENT_ROOT + '/' + os.listdir(EXPERIMENT_ROOT)[0] + '/' + 'seed_0')]
print(EXPERIMENT_ROOT)

# --- Initialize AestheticsValidator ---
print("Loading Simple Aesthetics Predictor...")
validator = AestheticsValidator()

# --- Utility Functions ---
def evaluate_image(image_path):
    try:
        return validator.evaluate_aesthetics(image_path)
    except Exception as e:
        print(f"[Error] {image_path}: {e}")
        return None

def parse_cfg_slg_from_column(column_name):
    """Parse CFG and SLG values from column names like 'slg_1.0_cfg_0.0'"""
    try:
        parts = column_name.split('_')
        slg_scale = float(parts[1])
        cfg_scale = float(parts[3])
        return cfg_scale, slg_scale
    except (IndexError, ValueError):
        return None, None

# --- Run Evaluation ---
results = []
print("Starting aesthetic evaluation...")

for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
    prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
    if not os.path.isdir(prompt_path):
        continue

    for seed_dir in os.listdir(prompt_path):
        seed_path = os.path.join(prompt_path, seed_dir)
        if not os.path.isdir(seed_path) or len(os.listdir(seed_path)) <len(IMAGE_TYPES):
            continue

        row = {'prompt': prompt_dir, 'seed': seed_dir}

        for mode in IMAGE_TYPES:
            img_path = os.path.join(seed_path, f"{mode}.png")

            if os.path.exists(img_path):
                score = evaluate_image(img_path)
                row[mode] = score
            else:
                row[mode] = None
                raise LookupError(f"Path not found: {img_path}")

        results.append(row)

# --- Save Results ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved results to {OUTPUT_CSV}")

# --- Print Stats ---
print("\n--- Overall Averages ---")
print(df[IMAGE_TYPES].mean())

print("\n--- Per Prompt Averages ---")
prompt_avg = df.groupby("prompt")[IMAGE_TYPES].mean()
print(prompt_avg)
prompt_avg.to_csv(f'{MODEL}_aesthetic_prompt_averages.csv')
if not multilayer_ablation:
    # --- Compute Overall Averages by CFG/SLG Configuration ---
    print("\n--- Computing CFG/SLG Configuration Averages ---")

    cfg_slg_mapping = {}
    for col in IMAGE_TYPES:
        cfg, slg = parse_cfg_slg_from_column(col)
        if cfg is not None and slg is not None:
            cfg_slg_mapping[col] = (cfg, slg)

    config_groups = defaultdict(list)
    for col, (cfg, slg) in cfg_slg_mapping.items():
        config_groups[(cfg, slg)].append(col)

    config_averages = []
    for (cfg, slg), columns in config_groups.items():
        config_scores = df[columns].mean(axis=1, skipna=True)
        overall_avg = config_scores.mean(skipna=True)

        config_averages.append({
            'cfg_scale': cfg,
            'slg_scale': slg,
            'avg_score': overall_avg
        })

    config_avg_df = pd.DataFrame(config_averages)
    config_avg_df = config_avg_df.sort_values(['cfg_scale', 'slg_scale'])

    config_csv = f'{MODEL}_aesthetic_cfg_slg_averages.csv'
    config_avg_df.to_csv(config_csv, index=False)
    print(f"Saved CFG/SLG configuration averages to {config_csv}")

    print("\n--- CFG/SLG Configuration Averages ---")
    print(config_avg_df)

    # Optional: Pivot table for easier visualization
    pivot_table = config_avg_df.pivot(index='slg_scale', columns='cfg_scale', values='avg_score')
    pivot_csv = f'{MODEL}_aesthetic_cfg_slg_pivot.csv'
    pivot_table.to_csv(pivot_csv)
    print(f"\nSaved pivot table to {pivot_csv}")
    print("\n--- Pivot Table (SLG x CFG) ---")
    print(pivot_table)


    # --- NEW: Compute and Display Averages by SLG Setting ---
    print("\n--- Computing SLG Setting Averages ---")

    slg_groups = defaultdict(list)
    for col, (cfg, slg) in cfg_slg_mapping.items():
        slg_groups[slg].append(col)

    slg_averages = []
    for slg_scale, columns in slg_groups.items():
        # Average across all CFG scales for this SLG setting
        slg_scores = df[columns].mean(axis=1, skipna=True)
        # Average across all prompts/seeds
        overall_avg = slg_scores.mean(skipna=True)
        
        slg_averages.append({
            'slg_scale': slg_scale,
            'avg_score': overall_avg,
            'num_configurations': len(columns)  # How many CFG scales for this SLG
        })

    slg_avg_df = pd.DataFrame(slg_averages)
    slg_avg_df = slg_avg_df.sort_values('slg_scale')

    slg_csv = f'{MODEL}_aesthetic_slg_averages.csv'
    slg_avg_df.to_csv(slg_csv, index=False)
    print(f"Saved SLG setting averages to {slg_csv}")

    print("\n--- Averages by SLG Setting ---")
    for _, row in slg_avg_df.iterrows():
        print(f"SLG {row['slg_scale']}: {row['avg_score']:.4f}")
