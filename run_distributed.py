#!/usr/bin/env python3
import torch
import torch.distributed as dist
import base64
from io import BytesIO
from diffusers import DiffusionPipeline
import PIL
import requests

sd = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)


def run_inference(rank,
                  world_size,
                  prompt:str="",
                  image_url:str="",
                  num_inference_steps:int=100,
                  image_guidance_scale:float=7.5,
                  guidance_scale:float=1.5,
                  pipe:DiffusionPipeline=sd):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    pipe.to(rank)

    image = download_image(image_url)
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps,
                       image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, image=image).images
    output = images[0]
    im_file = BytesIO()
    output.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return {"img":base64.b64encode(im_bytes), "gpu_index":rank}

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

