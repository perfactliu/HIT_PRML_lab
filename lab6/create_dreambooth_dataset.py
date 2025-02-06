from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5").to("cuda")
prompt = "a dog"
for i in range(100):
    cap = str(i).zfill(3)
    image = pipe(prompt).images[0]
    image.save("./dreambooth_dataset/"+cap+".jpg")
