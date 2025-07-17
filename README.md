# Contrastive Skip-Layer Guidance (CSLG)

This repository contains the official implementation of our paper:

**[Contrastive Skip-Layer Guidance for Controlling Semantic Coherence in Diffusion Models](https://arxiv.org/abs/...)**  
*Isabell Hans, Nikolai Röhrich — LMU Munich*


---

## Overview

Current diffusion models, including Stable Diffusion 3 and FLUX, often struggle with semantically complex prompts, like rendering visible, legible text, due to a trade-off between prompt adherence and image fidelity.

**Contrastive Skip-Layer Guidance (CSLG)** is a training-free, prompt-agnostic method for enhancing semantic coherence in such scenarios. It does so by:

1. Automatically identifying task-relevant layers using **contrastive prompt pairs**,
2. Selectively skipping these layers during inference,
3. Combining the result with standard Classifier-Free Guidance (CFG) for improved output quality at lower guidance scales.

<p align="center">
  <img src="assets/koala.drawio.png" width="600"/>
</p>

---

## Setup

### Requirements

- Python 3.9+
- diffusers
- OpenCV (for OCR-based evaluation)
- [FLUX](https://github.com/black-forest-labs/flux) or [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3)

Install dependencies:

```bash
pip install -r requirements.txt
```
---
## Usage
This section provides a step-by-step guide to using CSLG with diffusion models like FLUX or Stable Diffusion 3.
### 1. Run diffusion model on prompt pairs
Use pairs of prompts that differ only with respect to a specific semantic feature (e.g. presence of visible text as seen in prompt_datasets/prompts_text_notext.json) to generate images. 
Hook into the model's forward pass to get the intermediate activations per layer and save them as tensors.

```bash
# FLUX
python FLUX_LayerExperiment_pairwise.py

# Stable Diffusion 3
python SD3_LayerExperiment_pairwise.py
```

### 2. Identify Task-Relevant Layers
Run the contrastive analysis to compute cosine similarity differences between the activations of the two prompts for each layer. This will help identify which layers are most relevant for the task at hand.

```bash
python visualize.py # set MODEL to 'FLUX' or 'SD3' and specify the path to the saved activations
```
This creates a plot showing the relative cosine similarity for each layer, helping you identify which layers to skip during inference.

### 3. Apply CSLG during Inference
Using the identified layers, you can now apply CSLG during inference. To compare the results, we create images without any guidance, with standard Classifier-Free Guidance (CFG), and with CSLG skipping specified layers in all possible combinations.
```bash
#FLUX
python FLUX_final_experiments.py 

#Stable Diffusion 3
python SD3_final_experiments.py # specify the layers to skip in the script
```

---
## OCR-based Evaluation
To evaluate the effectiveness of CSLG in generating visible, legible text, we use an OCR-based approach. This involves running an OCR model on the generated images and comparing the results with the expected text extracted from the given prompts. 

```bash
python eval.py 
# set MODEL to 'FLUX' or 'SD3' and specify the path to the generated images, as well as the IMAGE_TYPES to be considered
```