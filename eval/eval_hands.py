""" 
This code evaluates the quality of hand generation in images by running MediaPipe hand detection and extracting the average hand detection confidence score.

It processes images from an experiment folder structured by prompts and ablation modes, computes average detection confidence per image,
then saves detailed CSV results and aggregate statistics.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import argparse
import sys

# --- HandValidator class ---
class HandValidator:
    def __init__(self, 
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe hand detection pipeline
        
        Args:
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking (ignored if static_image_mode=True)
        """
        self.mp_hands = mp.solutions.hands
        # Using static_image_mode=True as we are processing a batch of static images
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence # Ignored in static_image_mode
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, image_path: str) -> dict:
        """
        Detect hands in image and return only number of hands detected and average detection confidence.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with:
                - num_hands_detected: number of hands detected (int)
                - avg_detection_confidence: average detection confidence (float, 0.0 if no hands or error)
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image", "num_hands_detected": 0, "avg_detection_confidence": 0.0}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_handedness:
            return {"num_hands_detected": 0, "avg_detection_confidence": 0.0}

        # MediaPipe returns a classification score for each detected hand (left/right probability)
        # We use this score as the detection confidence for the purpose of quality evaluation.
        detection_scores = [hand_info.classification[0].score for hand_info in results.multi_handedness]
        num_hands = len(detection_scores)
        # Calculate the average score for all hands detected
        avg_score = float(np.mean(detection_scores)) if detection_scores else 0.0

        return {"num_hands_detected": num_hands, "avg_detection_confidence": avg_score}

# --- Utility Functions ---
def evaluate_image(image_path, validator):
    """ Helper function to call the validator and handle errors. """
    try:
        results = validator.detect_hands(image_path)
        # Return the average confidence score
        return results.get("avg_detection_confidence", 0.0)
    except Exception as e:
        print(f"[Error] {image_path}: {e}", file=sys.stderr)
        return None

def main(args):
    EXPERIMENT_ROOT = args.path
    MODEL = args.model
    OUTPUT_CSV = f'{MODEL}_hand_quality_ratings.csv'
    print(args.path)
    
    # --- Determine Image Types (e.g., ablation modes) ---
    first_prompt_dir = next((d for d in os.listdir(EXPERIMENT_ROOT) if os.path.isdir(os.path.join(EXPERIMENT_ROOT, d))), None)
    if not first_prompt_dir:
        raise FileNotFoundError(f"No prompt directories found in EXPERIMENT_ROOT: {EXPERIMENT_ROOT}")

    first_prompt_path = os.path.join(EXPERIMENT_ROOT, first_prompt_dir)
    IMAGE_TYPES = [filename[:-4] for filename in os.listdir(first_prompt_path) if filename.endswith('.jpg')]
    
    if not IMAGE_TYPES:
        raise FileNotFoundError(f"No .jpg files found in the first prompt directory: {first_prompt_path}")

    # --- Initialize HandValidator ---
    print("Loading MediaPipe HandValidator...")
    # Initialize with default confidence or allow configuration via arguments if desired
    validator = HandValidator(detection_confidence=args.detection_confidence) 

    # --- Run Evaluation ---
    results = []
    print("Starting hand evaluation...")

    for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
        prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
        if not os.path.isdir(prompt_path):
            continue
        row = {'prompt': prompt_dir}

        for mode in IMAGE_TYPES:
            img_path = os.path.join(prompt_path, f"{mode}.jpg")

            if os.path.exists(img_path):
                score = evaluate_image(img_path, validator)
                row[mode] = score
            else:
                row[mode] = None
                print(f"[Warning] Path not found: {img_path}")
                # You can uncomment the line below for a strict failure on missing files
                # raise LookupError(f"Path not found: {img_path}") 
        results.append(row)

    # --- Save Results ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")

    # --- Print Stats ---
    print("\n--- Overall Averages ---")
    print(df[IMAGE_TYPES].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hand quality in generated images using MediaPipe.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., FLUX) for output file naming.")
    parser.add_argument("--path", type=str, required=True, help="Root directory path for experiments (e.g., /path/to/layer_ablations).")
    parser.add_argument("--detection_confidence", type=float, default=0.7, help="Minimum confidence for MediaPipe hand detection.")
    args = parser.parse_args()
    main(args)