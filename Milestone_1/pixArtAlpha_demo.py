"""
PixArt-Alpha Code example from https://huggingface.co/docs/diffusers/en/api/pipelines/pixart
"""

import gc
import os
from pathlib import Path
import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5EncoderModel

remote_output_dir = os.environ.get("REMOTE_OUTPUT_DIR")
PROMPT = "cute cat"


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
    device_map="balanced"
)

#use pipeline to encode a prompt
with torch.no_grad():
    prompt = PROMPT
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)


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

latents = pipe(
    negative_prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    num_images_per_prompt=1,
    output_type="latent",
).images

del pipe.transformer
flush()

#decode into image
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]
os.makedirs(remote_output_dir, exist_ok=True)
file_path = os.path.join(remote_output_dir, f"{PROMPT}.png")
image.save(file_path)
print(f"Image saved at: {file_path}")