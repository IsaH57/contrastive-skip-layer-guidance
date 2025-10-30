import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import itertools
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1
import argparse

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
            
def evaluate_image(validator, image_path):
    try:
        return validator.evaluate_aesthetics(image_path)
    except Exception as e:
        print(f"[Error] {image_path}: {e}")
        return None

def main(args):
    EXPERIMENT_ROOT = args.path
    MODEL = args.model
    OUTPUT_CSV = f'{MODEL}_aesthetic_quality_ablations.csv'
    IMAGE_TYPES = [filename[:-4] for filename in os.listdir(EXPERIMENT_ROOT + '/' + os.listdir(EXPERIMENT_ROOT)[0])]
    print(EXPERIMENT_ROOT)

    validator = AestheticsValidator()
    results = []
    print("Starting aesthetic evaluation...")

    for prompt_dir in tqdm(os.listdir(EXPERIMENT_ROOT)):
        prompt_path = os.path.join(EXPERIMENT_ROOT, prompt_dir)
        if not os.path.isdir(prompt_path):
            continue
        row = {'prompt': prompt_dir}
        for mode in IMAGE_TYPES:
            img_path = os.path.join(prompt_path, f"{mode}.jpg")

            if os.path.exists(img_path):
                score = evaluate_image(validator, img_path)
                row[mode] = score
            else:
                row[mode] = None
                raise LookupError(f"Path not found: {img_path}")
        results.append(row)

    # --- Calculate, Save, and Print Averages ---
    df = pd.DataFrame(results)
    averages = df[IMAGE_TYPES].mean()
    averages.to_csv(OUTPUT_CSV, index_label='Mode', header=['Average_Score'])
    print(f"\nSaved average results to {OUTPUT_CSV}")

    # --- Print Stats ---
    print("\n--- Overall Averages ---")
    print(averages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., FLUX)")
    parser.add_argument("--path", type=str, required=True, help="Root directory path for experiments")
    args = parser.parse_args()
    main(args)