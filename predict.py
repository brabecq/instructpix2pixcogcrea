from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from typing import Any

import torch.multiprocessing as mp
from .run_distributed import run_inference

model_id = "timbrooks/instruct-pix2pix"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.predict,
            args=(world_size,),
            nprocs=world_size,
            join=False,
        )
        self.world_size = world_size

    # The arguments and types the model takes as input
    def predict(self, prompt:str="", image_url:str="", num_inference_steps:int=100, image_guidance_scale:float=20, guidance_scale:float=7.0, gpu_index=0) -> Any:
        """Run a single prediction on the model"""
        # create default process group
        return run_inference(gpu_index, self.world_size, prompt, image_url, num_inference_steps, image_guidance_scale, guidance_scale, self.pipe)