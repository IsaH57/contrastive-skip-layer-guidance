""" 
This code evaluates the quality of hand generation in images by running MediaPipe hand detection and extracting the average hand detection confidence score.

It processes images from an experiment folder structured by prompts and seeds, computes average detection confidence per image at different CFG and SLG guidance scales,
then saves detailed CSV results and aggregate statistics similar to the text visibility eval pipeline.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import itertools

layer_ablation = True
multilayer_ablation = False

# --- HandValidator class (simplified to only return hand count and avg detection confidence) ---
class HandValidator:
    def __init__(self, 
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe hand detection pipeline
        
        Args:
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, image_path: str) -> dict:
        """
        Detect hands in image and return only number of hands detected and average detection confidence.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with:
                - num_hands_detected: number of hands detected
                - avg_detection_confidence: average detection confidence (0 if no hands)
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image", "num_hands_detected": 0, "avg_detection_confidence": 0.0}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_handedness:
            return {"num_hands_detected": 0, "avg_detection_confidence": 0.0}

        detection_scores = [hand_info.classification[0].score for hand_info in results.multi_handedness]
        num_hands = len(detection_scores)
        avg_score = float(np.mean(detection_scores)) if detection_scores else 0.0

        return {"num_hands_detected": num_hands, "avg_detection_confidence": avg_score}

# --- Config ---
EXPERIMENT_ROOT = '/export/scratch/ru63zus/msg_images/experiments/flux_layer_ablation_hands'
MODEL = 'FLUX'
OUTPUT_CSV = f'{MODEL}_hand_quality_ratings.csv'

CFG_GUIDANCE_SCALES = [1., 2., 3., 4., 5.]
SLG_GUIDANCE_SCALES = [1., 2., 3., 4., 5.]
combinations = list(itertools.product(CFG_GUIDANCE_SCALES, SLG_GUIDANCE_SCALES))
IMAGE_TYPES = [f'slg_{slg_scale}_cfg_{cfg_scale}' for (cfg_scale, slg_scale) in combinations]

if layer_ablation: 
    IMAGE_TYPES = [f'layer_{i}' for i in range(19)]
if multilayer_ablation: 
    layers = [1,2,3,4,5]
    combinations = list(itertools.product(layers, SLG_GUIDANCE_SCALES))
    IMAGE_TYPES = [f'{i}_skipped_layers_slg_{slg_scale}' for (i, slg_scale) in combinations]
    IMAGE_TYPES.append('0_skipped_layers')

# --- Initialize HandValidator ---
print("Loading MediaPipe HandValidator...")
validator = HandValidator(detection_confidence=0.7)

# --- Utility Functions ---
def evaluate_image(image_path):
    try:
        results = validator.detect_hands(image_path)
        return results.get("avg_detection_confidence", 0.0)
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
print("Starting hand evaluation...")

for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
    prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
    if not os.path.isdir(prompt_path):
        continue

    for seed_dir in os.listdir(prompt_path):
        seed_path = os.path.join(prompt_path, seed_dir)
        if not os.path.isdir(seed_path) or len(os.listdir(seed_path)) < (19 if layer_ablation else len(combinations)+1):
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
prompt_avg.to_csv(f'{MODEL}_prompt_averages.csv')

if not layer_ablation and not multilayer_ablation:
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

    config_csv = f'{MODEL}_cfg_slg_averages.csv'
    config_avg_df.to_csv(config_csv, index=False)
    print(f"Saved CFG/SLG configuration averages to {config_csv}")

    print("\n--- CFG/SLG Configuration Averages ---")
    print(config_avg_df)

    # Optional: Pivot table for easier visualization
    pivot_table = config_avg_df.pivot(index='slg_scale', columns='cfg_scale', values='avg_score')
    pivot_csv = f'{MODEL}_cfg_slg_pivot.csv'
    pivot_table.to_csv(pivot_csv)
    print(f"\nSaved pivot table to {pivot_csv}")
    print("\n--- Pivot Table (SLG x CFG) ---")
    print(pivot_table)
