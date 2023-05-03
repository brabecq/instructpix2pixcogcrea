from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
import requests
from typing import Any

model_id = "timbrooks/instruct-pix2pix"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    # The arguments and types the model takes as input
    def predict(self, prompt:str="", image_url:str="", num_inference_steps:int=20, image_guidance_scale:float=20, guidance_scale:float=7) -> Any:
        """Run a single prediction on the model"""
        image = download_image(image_url)
        images = self.pipe(prompt=prompt, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images
        output = images[0]
        print("done")
        return output


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.resize((512, 512), resample=PIL.Image.NEAREST)
    image = image.convert("RGB")
    return image 

