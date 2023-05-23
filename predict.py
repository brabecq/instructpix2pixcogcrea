from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
import requests
from typing import Any
import base64
from io import BytesIO
from torch.nn import DataParallel


class Predictor(BasePredictor):
    def setup(self):
        # Define the device to run the model on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """Load the model into memory to make running multiple predictions efficient"""
        # model_id = "timbrooks/instruct-pix2pix"

        model = torch.load("checkpoints/instruct-pix2pix-00-22000.ckpt")
        # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
        #                                                                    safety_checker=None)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        self.model = model.to(self.device)
        # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @torch.inference_mode()
    def predict(self, prompt:str="", image_url:str="", num_inference_steps:int=100, image_guidance_scale:float=20, guidance_scale:float=7.0) -> Any:
        """Run a single prediction on the model"""
        image = download_image(image_url)
        images = self.model(prompt=prompt, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, image=image).images
        output = images[0]
        im_file = BytesIO()
        output.save(im_file, format="PNG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        return base64.b64encode(im_bytes)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image 


if __name__ == "__main__":
    Predictor().setup()
