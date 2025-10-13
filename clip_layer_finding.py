import os
import json
import torch
import gc
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import re
import random
import argparse
from FLUX_custom_pipeline import FluxPipeline

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    dataset = json.load(open(args.dataset_path, "r"))

    # Load Models
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    clip_scores = []

    for prompt_pair in dataset[:50]:
        print(f"Generating image for positive prompt: {prompt_pair['positive']}")
        images = []
        random_seed = random.randint(0, 2**32 - 1)

        for i in range(19):
            pipe.skipped_layers = [i]
            image = pipe(
                prompt_pair['positive'],
                negative_prompt=prompt_pair['positive'],
                num_inference_steps=28,
                max_sequence_length=256,
                true_cfg_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(random_seed)
            ).images[0]
            images.append(image)
        print(f"Calculating CLIP scores.")


        # Prepare inputs for CLIP
        positive_inputs = processor(
            text=[prompt_pair['positive']] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        negative_inputs = processor(
            text=[prompt_pair['negative']] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Calculate CLIP scores
        with torch.no_grad():
            positive_outputs = model(**positive_inputs)
            positive_scores = positive_outputs.logits_per_image.diagonal().cpu().tolist()
            
            negative_outputs = model(**negative_inputs)
            negative_scores = negative_outputs.logits_per_image.diagonal().cpu().tolist()
        
        # Store results
        result = {
            'positive_prompt': prompt_pair['positive'],
            'negative_prompt': prompt_pair['negative'],
            'positive_scores': positive_scores,
            'negative_scores': negative_scores,
            'avg_positive_score': sum(positive_scores) / len(positive_scores),
            'avg_negative_score': sum(negative_scores) / len(negative_scores),
            'seed': random_seed
        }
        
        clip_scores.append(result)
        
        print(f"Avg Positive Score: {result['avg_positive_score']:.4f}, Avg Negative Score: {result['avg_negative_score']:.4f}")
        
        # Clean up
        flush()

    # Save results
    output_path = f'clip_scores_{args.target}.json'
    with open(output_path, 'w') as f:
        json.dump(clip_scores, indent=2, fp=f)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with CLI arguments.")
    parser.add_argument("--target", type=str, required=True, help="Target concept. E.g. 'text', 'hands', etc.")
    parser.add_argument("--model", type=str, default="flux", help="Model type. flux or sd3.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON prompt file")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")

    args = parser.parse_args()
    main(args)