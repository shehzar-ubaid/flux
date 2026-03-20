import runpod
import torch
from diffusers import FluxPipeline
import io
import base64
import os

pipe = None

def load_model():
    global pipe
    if pipe is None:
        # GitHub Secrets se HF_TOKEN uthaye ga
        hf_token = os.environ.get("HF_TOKEN")
        
        # Direct Hugging Face se model load karega
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
            token=hf_token
        ).to("cuda")
    return pipe

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "A futuristic 4k avatar")
    num_images = job_input.get("num_images", 1)
    
    if num_images > 4: num_images = 4
    
    model = load_model()
    output_images = []

    for i in range(num_images):
        image = model(
            prompt,
            num_inference_steps=4,
            width=1024,
            height=1024
        ).images
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        output_images.append(img_str)
    
    return {"images": output_images}

runpod.serverless.start({"handler": handler})