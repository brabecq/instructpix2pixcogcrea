import base64
from io import BytesIO
from diffusers import DiffusionPipeline
import PIL
import requests
import logging


def run_inference(rank,
                  prompt: str = "",
                  image_url: str = "",
                  num_inference_steps: int = 100,
                  image_guidance_scale: float = 7.5,
                  guidance_scale: float = 1.5,
                  pipe: DiffusionPipeline = None):
    # Process the image
    image = download_image(image_url)
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps,
                  image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, image=image).images
    output = images[0]
    im_file = BytesIO()
    output.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    requests.post(f"http://localhost:8000/{rank}", json={"img": base64.b64encode(im_bytes).decode("utf-8")})
    # return {"img": base64.b64encode(im_bytes)}


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

