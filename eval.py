""" This code evaluates the visibility of text in images generated. It uses EasyOCR to extract text from images and compares it with the expected text from prompts."""
import os
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import easyocr
from difflib import SequenceMatcher

# --- Config ---
EXPERIMENT_ROOT = 'SD3_CFG_vs_SLG_experiments/sd3_results_20250710_002309'
MODEL = 'SD3'
OUTPUT_CSV = f'{MODEL}_visibility_ratings.csv'

IMAGE_TYPES = ['default_cfg', 'no_guidance', 'slg_skiplayer_9', 'slg_skiplayer_12', 'slg_skiplayer_9_12']

# --- Initialize EasyOCR ---
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

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

# --- Run Evaluation ---
results = []
print("Starting evaluation...")

for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
    prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
    if not os.path.isdir(prompt_path):
        continue

    for seed_dir in os.listdir(prompt_path):
        seed_path = os.path.join(prompt_path, seed_dir)
        if not os.path.isdir(seed_path) or len(os.listdir(seed_path)) < 3:
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