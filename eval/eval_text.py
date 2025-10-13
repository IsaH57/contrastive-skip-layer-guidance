""" This code evaluates the visibility of text in images generated. It uses EasyOCR to extract text from images and compares it with the expected text from prompts."""
import os
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import easyocr
from difflib import SequenceMatcher
import itertools
from collections import defaultdict

multilayer_ablation = True

# --- Config ---
EXPERIMENT_ROOT = '/export/home/ru63zus/repos/contrastive-skip-layer-guidance/experiments/flux_layer_ablation_simple_text'
print(EXPERIMENT_ROOT)
MODEL = 'FLUX'
OUTPUT_CSV = f'{MODEL}_visibility_ratings.csv'
IMAGE_TYPES = [filename[:-4] for filename in os.listdir(EXPERIMENT_ROOT + '/' + os.listdir(EXPERIMENT_ROOT)[0] + '/' + 'seed_0')]

# --- Initialize EasyOCR ---
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

# --- Utility Functions ---
def extract_quoted_text(prompt):
    matches = re.findall(r"'([^']+)'", prompt)
    return " ".join(matches) if matches else prompt  # fallback to full prompt if no matches

def compute_similarity_score(pred, target):
    ratio = SequenceMatcher(None, pred.lower(), target.lower()).ratio()
    return ratio

def extract_text(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def evaluate_image(image_path, prompt_text):
    try:
        visible_text = extract_quoted_text(prompt_text)
        image_text = extract_text(image_path)
        return compute_similarity_score(image_text, visible_text)
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
print("Starting evaluation...")

for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
    prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
    if not os.path.isdir(prompt_path):
        continue

    for seed_dir in os.listdir(prompt_path):
        seed_path = os.path.join(prompt_path, seed_dir)
        if not os.path.isdir(seed_path) or len(os.listdir(seed_path)) < len(IMAGE_TYPES):
            continue

        row = {'prompt': prompt_dir, 'seed': seed_dir}

        for mode in IMAGE_TYPES:
            img_path = os.path.join(seed_path, f"{mode}.png")

            if os.path.exists(img_path):
                score = evaluate_image(img_path, prompt_dir)
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
prompt_avg.to_csv(f'{MODEL}_prompt_averages.csv')

if not multilayer_ablation:
    # --- NEW: Compute Overall Averages by CFG/SLG Configuration ---
    print("\n--- Computing CFG/SLG Configuration Averages ---")

    # Parse CFG and SLG values from column names
    cfg_slg_mapping = {}
    for col in IMAGE_TYPES:
        cfg, slg = parse_cfg_slg_from_column(col)
        if cfg is not None and slg is not None:
            cfg_slg_mapping[col] = (cfg, slg)

    # Group columns by (CFG, SLG) configuration
    config_groups = defaultdict(list)
    for col, (cfg, slg) in cfg_slg_mapping.items():
        config_groups[(cfg, slg)].append(col)

    # Compute average scores for each configuration
    config_averages = []
    for (cfg, slg), columns in config_groups.items():
        # Average across all columns for this configuration (usually just one column)
        config_scores = df[columns].mean(axis=1, skipna=True)
        # Average across all prompts/seeds
        overall_avg = config_scores.mean(skipna=True)

        config_averages.append({
            'cfg_scale': cfg,
            'slg_scale': slg,
            'avg_score': overall_avg
        })

    # Convert to DataFrame and save
    config_avg_df = pd.DataFrame(config_averages)
    config_avg_df = config_avg_df.sort_values(['cfg_scale', 'slg_scale'])

    config_csv = f'{MODEL}_cfg_slg_averages.csv'
    config_avg_df.to_csv(config_csv, index=False)
    print(f"Saved CFG/SLG configuration averages to {config_csv}")

    print("\n--- CFG/SLG Configuration Averages ---")
    print(config_avg_df)

    # Optional: Create a pivot table for easier visualization
    pivot_table = config_avg_df.pivot(index='slg_scale', columns='cfg_scale', values='avg_score')
    pivot_csv = f'{MODEL}_cfg_slg_pivot.csv'
    pivot_table.to_csv(pivot_csv)
    print(f"\nSaved pivot table to {pivot_csv}")
    print("\n--- Pivot Table (SLG x CFG) ---")
    print(pivot_table)