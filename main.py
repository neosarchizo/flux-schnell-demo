import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-schnell', 
    revision='refs/pr/1', 
    torch_dtype=torch.bfloat16
).to('mps')

prompt = "A cat holding a sign that says hello world"

out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=1024,
    width=1024,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]

out.save('out.png')