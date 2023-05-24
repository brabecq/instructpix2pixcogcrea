from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from typing import Any
import base64
from io import BytesIO
from diffusers import DiffusionPipeline
import PIL
import requests
import torch.distributed as dist
import logging

from torch.multiprocessing import Process
model_id = "timbrooks/instruct-pix2pix"

def run_inference(rank,
                  prompt: str = "",
                  image_url: str = "",
                  num_inference_steps: int = 100,
                  image_guidance_scale: float = 7.5,
                  guidance_scale: float = 1.5,
                  pipe: DiffusionPipeline = None):
    # Log the process
    log_process = "Processing job_id: %s, job_operator: %s \n" % (rank)
    logging.info(log_process)

    # Process the image
    image = download_image(image_url)
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps,
                  image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, image=image).images
    output = images[0]
    im_file = BytesIO()
    output.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return {"img": base64.b64encode(im_bytes), "gpu_index": rank}


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.world_size = torch.cuda.device_count()
        self.pipe = [StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
                     .to(f"cuda:{i}")
                        for i in range(self.world_size)]
        # self.pipe.schedulers = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    # The arguments and types the model takes as input
    @torch.inference_mode()
    def predict(self,
                prompt:str="",
                image_url:str="",
                num_inference_steps:int=100,
                image_guidance_scale:float=20,
                guidance_scale:float=7.0,
                gpu_index:int=0
                ) -> Any:
        """Run a single prediction on the model"""

        task = Process(target=run_inference, args=(gpu_index,
                             prompt, image_url,
                             num_inference_steps,
                             image_guidance_scale,
                             guidance_scale,
                             self.pipe[gpu_index]))
        task.start()
        logging.log(logging.INFO, "Started task on GPU %s" % gpu_index)