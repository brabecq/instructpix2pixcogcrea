from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from run_inference import run_inference
from typing import Any
import logging
import requests

from torch.multiprocessing import Value


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
                id: int = 0
                ) -> Any:
        """Run a single prediction on the model"""
        task = Value(target=run_inference, args=(id,
                             prompt, image_url,
                             num_inference_steps,
                             image_guidance_scale,
                             guidance_scale,
                             self.pipe[id]))
        task.start()
        print("Started task on GPU %s" % id)
        # result = run_inference(id,
        #               prompt, image_url, num_inference_steps,
        #                 image_guidance_scale, guidance_scale, self.pipe[id])
        result = task.get()
        print("Task finished")
        return result