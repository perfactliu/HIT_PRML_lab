# 原模型輸出

# from diffusers import DiffusionPipeline
# pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5").to("cuda")
# prompt = "a cat"
# image = pipe(prompt).images[0]
# image.save("a_cat_pre_2.png")



# lora模型輸出

# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.unet.load_attn_procs('fine_tuned_lora_unet')
# pipe.to("cuda")
# image = pipe("a cat", num_inference_steps=25).images[0]
# image.save("a_cat_aft_3.png")



# dreambooth模型輸出

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs('fine_tuned_dreambooth_unet')
pipe.to("cuda")
image = pipe("a sks dog eating", num_inference_steps=25).images[0]
image.save("a_sks_dog_eat.png")





