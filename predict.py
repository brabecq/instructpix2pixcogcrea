from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from run_inference import run_inference
from typing import Any
import logging

from torch.multiprocessing import Process


class Predictor(BasePredictor):
    def setup(self):
        model_id = "timbrooks/instruct-pix2pix"
        """Load the model into memory to make running multiple predictions efficient"""
        self.world_size = torch.cuda.device_count()
        self.pipe = [
            StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None).to(f"cuda:{i}")
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