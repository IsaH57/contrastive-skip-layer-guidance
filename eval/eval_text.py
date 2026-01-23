""" This code evaluates the visibility of text in images generated. It uses EasyOCR to extract text from images and compares it with the expected text from prompts."""
import os
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import easyocr
from difflib import SequenceMatcher
import itertools
import json
from collections import defaultdict
import argparse

# --- Utility Functions ---
def extract_quoted_text(prompt):
    matches = re.findall(r"'([^']+)'", prompt)
    return " ".join(matches) if matches else prompt  # fallback to full prompt if no matches

def compute_similarity_score(pred, target):
    ratio = SequenceMatcher(None, pred.lower(), target.lower()).ratio()
    return ratio

def extract_text(image_path, reader):
    # detail=0 returns only the recognized text strings
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def evaluate_image(image_path, prompt, reader):
    try:
        # The expected text to be visible (quoted part of the prompt)
        visible_text = extract_quoted_text(prompt)
        
        # Text extracted by EasyOCR
        image_text = extract_text(image_path, reader)
        
        # Compute the similarity score
        return compute_similarity_score(image_text, visible_text)
    except Exception as e:
        print(f"[Error] {image_path}: {e}")
        return None

def main(args):
    print(args.path)
    EXPERIMENT_ROOT = args.path
    MODEL = args.model
    OUTPUT_CSV = f'{MODEL}_visibility_ratings.csv'
    
    # --- Determine Image Types (e.g., ablation modes) ---
    # Assumes the first directory in EXPERIMENT_ROOT contains images
    first_prompt_dir = os.listdir(EXPERIMENT_ROOT)[1]
    first_prompt_path = os.path.join(EXPERIMENT_ROOT, first_prompt_dir)
    print(first_prompt_path)

    if os.path.isdir(first_prompt_path):
        # Extract image types from filenames (e.g., 'base.jpg' -> 'base')
        IMAGE_TYPES = [f[:-4] for f in os.listdir(EXPERIMENT_ROOT + '/' + [d for d in os.listdir(EXPERIMENT_ROOT) if not d.endswith('.json')][0]) if not f.endswith('.json')]    
    else:
        # Fallback if the first item isn't a directory (shouldn't happen with correct structure)
        raise FileNotFoundError(f"Could not find experiment structure in {EXPERIMENT_ROOT}")

    # --- Initialize EasyOCR ---
    # Using 'en' for English and gpu=True for acceleration
    print("Loading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    # --- Load Prompt Data ---
    # Assumes 'results_log.json' contains a map of prompt_dir_name to the full prompt string
    log_path = os.path.join(EXPERIMENT_ROOT, 'results_log.json')
    if not os.path.exists(log_path):
         raise FileNotFoundError(f"Required file not found: {log_path}")
         
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    print("Starting evaluation...")

    # --- Run Evaluation ---
    # Iterate through each prompt's directory
    for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
        prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
        
        if not os.path.isdir(prompt_path):
            continue

        # Get prompt text from the loaded data
        if prompt_dir not in data:
            print(f"[Warning] Prompt data not found for directory: {prompt_dir}. Skipping.")
            continue
            
        prompt = data[prompt_dir]
        row = {'prompt': prompt}

        # Iterate through each image type (ablation mode) for the current prompt
        for mode in IMAGE_TYPES:
            img_path = os.path.join(prompt_path, f"{mode}.jpg")
            
            if os.path.exists(img_path):
                # Evaluate the image
                score = evaluate_image(img_path, prompt, reader)
                row[mode] = score
            else:
                row[mode] = None
                # Raise an error if an expected image file is missing
                # raise LookupError(f"Path not found: {img_path}") # Commented out for smoother run, uncomment for strict check
                print(f"[Warning] Path not found: {img_path}")

        results.append(row)

    # --- Save Results ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")

    # --- Print Stats ---
    print("\n--- Overall Averages ---")
    print(df[IMAGE_TYPES].mean())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text visibility in generated images using EasyOCR.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., FLUX) for output file naming.")
    parser.add_argument("--path", type=str, required=True, help="Root directory path for experiments (e.g., /path/to/layer_ablations).")
    args = parser.parse_args()
    main(args)