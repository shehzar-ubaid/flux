FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# AI aur RunPod libraries install karein
RUN pip3 install --no-cache-dir torch diffusers transformers accelerate safetensors Pillow runpod

# Handler copy karein
COPY handler.py .

# Model download karne ki command (RunPod start hote hi download kar lega)
# Is se GitHub par load nahi paray ga
ENV HF_HUB_ENABLE_HF_TRANSFER=1
CMD ["python3", "-u", "handler.py"]