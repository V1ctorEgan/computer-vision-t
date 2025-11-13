import sys

import diffusers
import matplotlib.pyplot as plt
import torch
import transformers
from IPython.display import display
from PIL import Image
from torchinfo import summary
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using {device} device with {dtype} data type.")

tokenizer = transformers.CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
    torch_dtype=dtype
)

print(tokenizer)

# task 1
text = "Hello, world!"
result = tokenizer(text)

print(type(result))
print(result)

# task 2
for token in result.input_ids:
    print(tokenizer.decode(token))

prompt = "A red bird flies through a blue sky over a green tree."
text_tokens = tokenizer(
    prompt,
    padding="max_length",  # Give us as many tokens as the model can handle.
    truncation=True,  # Truncate the output if it would give us more tokens.
    return_tensors="pt",  # Return a PyTorch tensor.
)

print(text_tokens.input_ids)
print(text_tokens.input_ids.shape)

# task 3
uncond_tokens = tokenizer(
    "",
    padding="max_length",  # Give us as many tokens as the model can handle.
    truncation=True,  # Truncate the output if it would give us more tokens.
    return_tensors="pt", 
)

print(uncond_tokens.input_ids)
print(uncond_tokens.input_ids.shape)

# task 4
# we need to turn the words into embeddings which the model( stable diffusion) can understand
embedder = transformers.CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
    torch_dtype=dtype
)
embedder.to(device)  # Do this on the GPU

# Print out a summary of this neural network
summary(embedder)

with torch.no_grad():  # No need for gradient calculations
    text_embedding = embedder(text_tokens.input_ids.to(device))

print(type(text_embedding))
print(text_embedding.keys())

# task 4
print("Class:", type(text_embedding.last_hidden_state))
print("Shape:", text_embedding.last_hidden_state.shape)

# task 5
with torch.no_grad():
    uncond_embedding = embedder(uncond_tokens.input_ids.to(device))

print(uncond_embedding.last_hidden_state.shape)

# task 6
all_embeddings = torch.cat([uncond_embedding.last_hidden_state,text_embedding.last_hidden_state])

print(all_embeddings.shape)


vae = diffusers.AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae",
    torch_dtype=dtype
)
vae.to(device)  # Run it on the GPU

summary(vae)

height = 512
width = 512
scale_factor = 8
n_channels = vae.config.latent_channels
print(n_channels)
# Latent vectors will be a 4-D tensor, representing (batch, channel, height, width)

# task 7
latent_shape = (
    1, # Batch size
    n_channels, # Latent channels
    height // scale_factor, # Height
   width // scale_factor, # Width
)

random_latents = torch.randn(latent_shape, device=device, dtype=dtype)

print(random_latents.shape)

# task 8
def plot_latents(latents):
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(latents[0, i].cpu().numpy())
        plt.colorbar()

# Plot the latent vector
plot_latents(random_latents)

# task 9
latents = random_latents
# Decode the latents
with torch.no_grad():
    scaling_factor = torch.tensor(vae.config.scaling_factor, device=device, dtype=dtype)
    decoded = vae.decode(latents / scaling_factor).sample

print("Shape:", decoded.shape)


# This is in (batch, channels, height, width) format. However, PIL expects images in (height, width, 
# channels) format. We can use the .permute() method to switch things around. We'll also convert the tensor
#  into a NumPy array.
# Permute the dimensions and convert to NumPy
unscaled = decoded.permute(0, 2, 3, 1).cpu().numpy()
print("Shape:", unscaled.shape)
# task 10
plt.hist(
    unscaled.flatten(),
    bins=50,
)
# Scale the image values to be between 0 and 255
scaled = ((unscaled + 1) * 255 / 2).clip(0, 255).round().astype("uint8")


# task 11
# Create a PIL image
Image.fromarray(scaled[0])

# task 12
def latents_to_image(latents, vae=vae):
    """Transform the latent vector to a image, using a VAE decoder.

    Inputs:  latents  Latent vector(s) as a 4-D PyTorch tensor.  Only
                      the first element of the batch will be used.
             vae      The VAE used to decode the image from latents.

    Outputs: A PIL image corresponding to the latents.
    """
    # Scaling factor
    scaling_factor = torch.tensor(vae.config.scaling_factor, device=device, dtype=dtype)
    
    # Decode the latents
    with torch.no_grad():
        decoded = vae.decode(latents / scaling_factor).sample
    # Permute the dimensions and convert to NumPy
    unscaled = decoded.permute(0, 2, 3, 1).cpu().numpy()
    # Scale the image values to be between 0 and 255
    scaled = ((unscaled + 1) * 255 / 2).clip(0, 255).round().astype("uint8")
    # Return a PIL image
    return Image.fromarray(scaled[0])

latents_to_image(random_latents)


unet = diffusers.UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet",
    torch_dtype=torch.float16
)
unet.to(device)

summary(unet)

scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)

print(scheduler)
# task 13
latents = random_latents * scheduler.init_noise_sigma

latents.shape

# task 14
t =  scheduler.timesteps[0]

print(t)

# task 15
# Assemble the latent inputs
latent_inputs = torch.cat([latents, latents])

print(latent_inputs.shape)

# Predict the noise
scaled_inputs = scheduler.scale_model_input(latent_inputs, timestep=t)
with torch.no_grad():
    noise = unet(scaled_inputs, t, encoder_hidden_states=all_embeddings)

# Split the unconditioned and conditioned predictions
noise_uncond, noise_cond = noise.sample.chunk(2)

# task 16