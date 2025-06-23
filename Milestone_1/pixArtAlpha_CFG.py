"""
PixArt-Alpha Code example from https://huggingface.co/docs/diffusers/en/api/pipelines/pixart
"""

import gc
import inspect
import os
from pathlib import Path
import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5EncoderModel


current_file = inspect.getfile(inspect.currentframe())
file_path = os.path.splitext(os.path.basename(current_file))[0]

output_dir = os.path.join(os.getcwd(), f"img_{file_path}")
os.makedirs(output_dir, exist_ok=True)

PROMPT = "an ancient temple in the jungle, covered in moss, with golden statues and shafts of light piercing through the trees"

GUIDANCE_SCALE = 7.5


#load text encoder
text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="text_encoder",

    load_in_4bit=True,
    device_map="balanced",

)

pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="balanced",
)

#use pipeline to encode a prompt
with torch.no_grad():
    prompt = PROMPT
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
# prompt_embeds: conditioned prompt
# negative_prompt_embed: unconditioned prompt

#remove encoder and pipe from the memory
def flush():
    gc.collect()
    torch.cuda.empty_cache()

del text_encoder
del pipe
flush()

#compute latents with prompt embedding as input
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")


#generate images with different guidance scale
for gs in [1.0, 3.0, 5.0, 7.5, 12.0]:
    print(f"Generating with CFG scale: {gs}")
    latents = pipe(
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        num_images_per_prompt=1,
        output_type="latent",
        guidance_scale=gs,
    ).images

    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    image.save(os.path.join(output_dir, f"temple_cfg{gs}.png"))
