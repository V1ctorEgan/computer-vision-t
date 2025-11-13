import os
import sys

import datasets
import diffusers
import huggingface_hub
import requests
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from IPython.display import display

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using {device} device with {dtype} data type.")

# task 1
MODEL_NAME = "CompVis/stable-diffusion-v1-4"

pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
    MODEL_NAME, torch_dtype=dtype
)
pipeline.to(device)

print(type(pipeline))

# task 2
images = pipeline(["My dog Maya"] * 4)
for im in images:
    display(im)

# task 3
DATASET_NAME = "worldquant-university/maya-dataset-v1"
data_builder = datasets.load_dataset_builder(DATASET_NAME)

print(data_builder.dataset_name)

# task 4
print(data_builder.info.splits)

# task 5
data = datasets.load_dataset(DATASET_NAME)

print(data)

# task 6
# The values are PIL images, so they will be displayed
# automatically by Jupyter.
data["train"]["image"][3]

# task 7
# Use dictionary indexing to look up the text values.
data["train"]["text"]

# task 8
OUTPUT_DIR = "maya_model_v1_lora"


# task 9 tolarge to run
# task 10
pipeline.load_lora_weights(
      OUTPUT_DIR, # Directory containing weights file

    weight_name="pytorch_lora_weights.10_epochs.safetensors",
)
# task 11
images = pipeline(["My dog Maya"] * 4).images

for im in images:
    display(im)