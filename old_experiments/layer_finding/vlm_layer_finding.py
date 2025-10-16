import os
import json
import torch
import gc
from PIL import Image
from pathlib import Path
import re
import argparse
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_evaluation_prompts(target):
    """
    Returns target-specific evaluation prompts.
    
    Args:
        target: The evaluation target (e.g., 'text', 'hands', 'aesthetics')
    
    Returns:
        tuple: (description_prompt, scoring_prompt_template)
    """
    prompts = {
        'text': {
            'description': '''<image>
            CRITICAL TASK: Describe every single bit of text that is visible in the image in detail. List all text exactly as it appears, character by character.''',

            'scoring': '''The image was generated with this prompt: "{generation_prompt}"
            Based on your previous description of the text in the image, now evaluate the quality of the generated text in the image.
    
            Compare your description of visible text versus what text was expected. Check for:
            - Wrong letters
            - Missing or extra letters
            - Completely hallucinated text that shouldn't be there
            - Spelling errors
            - Any other text defects
            - ...

            Use VERY STRICT scoring criteria.
            End with: "SCORE: X/10"'''
        },

        'hands': {
            'description': '''<image>
            CRITICAL TASK: Describe every single human hand visible in the image in great detail. For each hand, describe:
            - Number of fingers
            - Position and pose of the hand
            - Any anatomical issues or abnormalities
            - Whether fingers look natural and properly formed
            - Connection to arms/wrists,
            - ...''',

            'scoring': '''
            Based on your previous description of the hands in the image, now critically evaluate the quality of the hands in the image.

            Evaluate based on:
            - General visibility of hands
            - Correct number of fingers (should be 5 per hand if hand is fully visible)
            - Natural finger proportions and positions
            - Proper hand anatomy
            - Realistic joints and connections
            - No extra or missing fingers
            - No distorted or malformed fingers
            - Natural hand poses
            - ...

            Use VERY STRICT scoring criteria.
            End with: "SCORE: X/10"'''
        },

        'aesthetics': {
            'description': '''<image>
            CRITICAL TASK: Describe the image in great detail. In your description, include all properties of the image that increase or decrease its aesthetic value and their relevance for its overall aesthetic value. Be very critical. Include:
            - Overall image quality 
            - Composition and framing
            - Color harmony and palette
            - Lighting quality
            - Visual appeal and artistic merit
            - Technical quality (sharpness, exposure, etc.)
            - Overall mood and atmosphere
            - ...''',
            'scoring': '''
            Based on your previous description of the image's aesthetic qualities, now provide an overall aesthetic score. 
            Make sure that images that are beautiful, but not outstandingly aesthetic, do not get high scores. 

            Use VERY STRICT scoring criteria. 
            End with: "SCORE: X/10"'''
        }
    }
    
    if target not in prompts:
        raise ValueError(f"Unknown target: {target}. Available targets: {list(prompts.keys())}")
    
    return prompts[target]['description'], prompts[target]['scoring']

def evaluate_quality_two_step(image_path, generation_prompt, target, tokenizer, model, load_image):
    """
    Evaluate image quality using a two-step process:
    1. First get detailed description of the target aspect
    2. Then score based on that description
    
    Args:
        image_path: Path to the image file
        generation_prompt: The prompt used to generate the image
        target: What to evaluate (e.g., 'text', 'hands', 'aesthetics')
        tokenizer: Model tokenizer
        model: Vision-language model
        load_image: Image loading function
    
    Returns:
        tuple: (score, description, scoring_response)
    """
    description_prompt, scoring_template = get_evaluation_prompts(target)
    
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    # STEP 1: Get detailed description
    description, history = model.chat(
        tokenizer, 
        pixel_values, 
        description_prompt, 
        generation_config,
        history=None, 
        return_history=True
    )
    
    # STEP 2: Score based on the description
    scoring_prompt = scoring_template.format(generation_prompt=generation_prompt)
    
    scoring_response, _ = model.chat(
        tokenizer, 
        pixel_values, 
        scoring_prompt, 
        generation_config,
        history=history,
        return_history=True
    )
    
    # Extract score - handles both integer and decimal scores
    score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)/10', scoring_response, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else None
    
    return score, description, scoring_response

def create_prompt_mapping(prompts_list):
    """Create mapping from directory name (first 20 chars) to full prompt"""
    mapping = {}
    for prompt in prompts_list:
        # Handle both string prompts and dict prompts
        if isinstance(prompt, str):
            full_prompt = prompt
        elif isinstance(prompt, dict):
            full_prompt = prompt.get('positive', prompt.get('prompt', ''))
        else:
            full_prompt = str(prompt)
        
        # Create directory name (first 20 chars)
        dir_name = full_prompt[:20]
        mapping[dir_name] = full_prompt
    
    return mapping

def main(args):
    # Load prompts from JSON file
    print(f"Loading prompts from: {args.prompts_path}")
    with open(args.prompts_path, 'r') as f:
        prompts_list = json.load(f)
    
    # Create mapping from directory name to full prompt
    prompt_mapping = create_prompt_mapping(prompts_list)
    print(f"Loaded {len(prompt_mapping)} prompts")
    
    # Load InternVL3 Model
    print("Loading InternVL3 model...")
    model_path = args.internvl_model_path or "OpenGVLab/InternVL3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()

    quality_results = []

    description_prompt, scoring_template = get_evaluation_prompts(args.target)
    print(f'Performing Evaluation for images in: {args.image_root}')
    print('Using description prompt:')
    print(description_prompt)
    print('Using evaluation prompt:')
    print(scoring_template)

    # Get all prompt directories
    prompt_dirs = [d for d in os.listdir(args.image_root) 
                   if os.path.isdir(os.path.join(args.image_root, d))]
    
    print(f"\nFound {len(prompt_dirs)} prompt directories")

    for prompt_dir in prompt_dirs[:10]:
        prompt_path = os.path.join(args.image_root, prompt_dir)
        
        # Get full prompt from mapping
        if prompt_dir not in prompt_mapping:
            print(f"Warning: No prompt found for directory '{prompt_dir}', skipping...")
            continue
        
        prompt = prompt_mapping[prompt_dir]
        
        print(f"\n{'='*80}")
        print(f"Processing prompt: {prompt}")
        print(f"{'='*80}")
        
        prompt_results = {
            'prompt': prompt,
            'prompt_dir': prompt_dir,
            'target': args.target,
            'seeds': []
        }
        
        # Get all seed directories
        seed_dirs = [d for d in os.listdir(prompt_path) 
                     if os.path.isdir(os.path.join(prompt_path, d)) and d.startswith('seed_')]
        
        for seed_dir in seed_dirs:
            
            seed_path = os.path.join(prompt_path, seed_dir)
            
            # Extract seed number
            seed_num = seed_dir.replace('seed_', '')
            
            seed_result = {
                'seed': seed_num,
                'layers': []
            }
            print(f'Processing seed {seed_num}')
            # Process each layer image
            for layer_idx in range(19):
                image_path = os.path.join(seed_path, f'layer_{layer_idx}.png')
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                                
                # Evaluate quality with retry until valid score
                max_retries = 10
                retry_count = 0
                score = None
                
                while score is None and retry_count < max_retries:
                    try:
                        score, description, scoring_response = evaluate_quality_two_step(
                            image_path=image_path,
                            generation_prompt=prompt,
                            target=args.target,
                            tokenizer=tokenizer,
                            model=model,
                            load_image=load_image
                        )
                        
                        if score is not None:
                            layer_result = {
                                'layer': layer_idx,
                                'score': score,
                                'image_path': image_path
                            }
                            seed_result['layers'].append(layer_result)
                            print(f"Layer {layer_idx} Score: {score}/10 (attempts: {retry_count + 1})")
                        else:
                            retry_count += 1
                            print(f"No score extracted, retrying... (attempt {retry_count}/{max_retries})")
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"Error evaluating layer {layer_idx} (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count >= max_retries:
                            layer_result = {
                                'layer': layer_idx,
                                'score': None,
                                'image_path': image_path
                            }
                            seed_result['layers'].append(layer_result)
                            print(f"Max retries reached for layer {layer_idx}")
                
                if score is None and retry_count >= max_retries:
                    print(f"Failed to get valid score for layer {layer_idx} after {max_retries} attempts")
                
                flush()
            
            # Calculate average score for this seed
            valid_scores = [l['score'] for l in seed_result['layers'] if l['score'] is not None]
            seed_result['avg_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else None
            
            prompt_results['seeds'].append(seed_result)
            
            if seed_result['avg_score'] is not None:
                print(f"Average score for {seed_dir}: {seed_result['avg_score']:.2f}/10")
        
        # Calculate overall statistics across all seeds
        all_scores = [s['avg_score'] for s in prompt_results['seeds'] if s['avg_score'] is not None]
        prompt_results['overall_avg_score'] = sum(all_scores) / len(all_scores) if all_scores else None
        
        quality_results.append(prompt_results)
        
        if prompt_results['overall_avg_score'] is not None:
            print(f"\nOverall average score for this prompt: {prompt_results['overall_avg_score']:.2f}/10")
        
    # Save intermediate results
    output_path = f'{args.target}_quality_scores.json'
    with open(output_path, 'w') as f:
        json.dump(quality_results, f, indent=2)
    
    flush()

    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-generated images using InternVL3")
    parser.add_argument("--target", type=str, required=True, 
                        choices=['text', 'hands', 'aesthetics'],
                        help="Target aspect to evaluate")
    parser.add_argument("--image_root", type=str, required=True, 
                        help="Root directory containing prompt dirs with seed subdirs and layer images")
    parser.add_argument("--prompts_path", type=str, required=True,
                        help="Path to JSON file containing the full prompts")
    parser.add_argument("--internvl_model_path", type=str, default="OpenGVLab/InternVL3-14B", 
                        help="Path to InternVL3 model")
    
    args = parser.parse_args()
    main(args)