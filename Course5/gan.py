import datetime
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import make_grid
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")

# task 1
data_dir = Path("data_p5")
images_dir = data_dir / "gan_training_images"
print(images_dir)

# task 2
IMAGE_SIZE = 64

transformations = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

transformations

# task 3
batch_size = 128

dataset = datasets.ImageFolder(root=data_dir, transform=transformations)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

single_batch = next(iter(dataloader))[0]
print(f"Batches have shape: {single_batch.shape}")
# code before 4
discriminator = nn.Sequential()
discriminator.append(nn.Flatten())

# Input images are 1 * 64 * 64 = 4096 pixels after flattening
discriminator.append(nn.Linear(1 * 64 * 64, 1024))
discriminator.append(nn.LeakyReLU(0.25))

discriminator(single_batch).shape
# task 4
# Add linear layer with 512 and Leaky ReLU
discriminator.append(nn.Linear(1024, 512))
discriminator.append(nn.LeakyReLU(0.25))
# Add linear layer with 256 and Leaky ReLU
discriminator.append(nn.Linear(512, 256))
discriminator.append(nn.LeakyReLU(0.25))


discriminator

# task 5
discriminator.append(nn.Linear(256, 1))

print("Getting results with following shape:")
print(discriminator(single_batch).shape)
discriminator.append(nn.Sigmoid())

# // the discriminator has been completed now time for the generator:

noise_size = 100
first_stage_size = 256

generator = nn.Sequential()
generator.append(nn.Linear(noise_size, first_stage_size, bias=False))

# The batch norm doesn't change the shape, but needs the number of inputs
# The 0.8 adjusts its behavior, we'll use the same value for all stages
generator.append(nn.BatchNorm1d(first_stage_size, 0.8))
generator.append(nn.LeakyReLU(0.25))

# Create a batch of random numbers
random_number_sample = torch.randn(batch_size, noise_size)
random_number_sample.shape

# task 6
first_upscale = generator(random_number_sample)

print(f"After first upscale: {first_upscale.shape}")

# task 7
second_stage_size = 512

# Add second upsampling stage
generator.append(nn.Linear(first_stage_size,second_stage_size , bias=False))


generator.append(nn.BatchNorm1d(second_stage_size, 0.8))
generator.append(nn.LeakyReLU(0.25))

# task 8
third_stage_size = 1024

# Add third upsampling stage
generator.append(nn.Linear(second_stage_size,third_stage_size , bias=False))
generator.append(nn.BatchNorm1d(third_stage_size, 0.8))
generator.append(nn.LeakyReLU(0.25))

# task 9
output_size = 1 * 64 * 64
# Add the final linear layer
generator.append(nn.Linear(third_stage_size,output_size, bias=False))

generator

generator.append(nn.Tanh())
generator.append(nn.Unflatten(1, [1, 64, 64]))
# task 10
generator_output = generator(random_number_sample)
output_shape = generator_output.shape

print(f"Generator output shape: {output_shape}")

# That's our generator!

# Right now it's untrained, so it will create images that are just noise. Here's what one looks like.

plt.imshow(generator_output[0, 0].detach(), cmap="gray")
plt.axis("off")

# Gets the time in the order year-month-day_hour-minute-second
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
now

# task 11
runs_dir = Path("runs")
now_dir = runs_dir / now

# Create the directories with mkdir
now_dir.mkdir(parents=True)
print(f'directory "{now_dir}" exists: {now_dir.exists()}')

# task 12
lr = 0.0002
betas = (0.5, 0.999)

disc_opt = torch.optim.AdamW(params=discriminator.parameters(), lr=lr, betas=betas)
gen_opt = torch.optim.AdamW(params=generator.parameters(), lr=lr, betas=betas)

n_disc_pars = len(disc_opt.param_groups[0]["params"])
n_gen_pars = len(gen_opt.param_groups[0]["params"])

print(f"disc_opt sees {n_disc_pars} parameters, should be 8")
print(f"get_opt sees {n_gen_pars} parameters, should be 10")


loss_function = nn.BCELoss()
#  information to be read in the gan.txt

# task 13
# discriminator to device
discriminator.to(device)
# generator to device
generator.to(device)
disc_dev = next(generator.parameters()).device.type
gen_dev = next(generator.parameters()).device.type

print(f"Discriminator on {disc_dev}")
print(f"Generator on {gen_dev}")

# task 14
def make_random_images(batch_size, generator=generator):
    # Create a batch of random numbers
    random_number_sample = torch.randn(batch_size, noise_size)
    random_number_sample = random_number_sample.to(device)
    generator.eval()

    # Run the generator on the random numbers
    generator_output = generator(random_number_sample)
    return generator_output

# Test run
sample_images = make_random_images(batch_size, generator)
print(f"Output shape: {sample_images.shape}")

# display
def display_images(image_tensor, n_to_display=6):
    grid = make_grid(image_tensor[:n_to_display], nrow=6, normalize=True)
    img_out = transforms.ToPILImage()(grid)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(img_out)
    plt.axis("off")
    plt.show()

display_images(sample_images)

def perform_batch_step(
    discriminator,
    generator,
    real_data_batch,
    loss_function,
    disc_opt,
    gen_opt,
    device=device,
):
    """Perform a single batch step"""

    # Set real and fake labels
    real_label_val = 1.0
    fake_label_val = 0.0

    # Send real data to device
    # This pulls out just the images, we'll make our own label
    real_images = real_data_batch[0].to(device)

    # Create labels: all 1.0 for real data
    actual_batch_size = real_images.size(0)
    real_label = torch.full((actual_batch_size, 1), real_label_val, device=device)

    # Get the derivative for the real images
    disc_opt.zero_grad()
    real_output = discriminator(real_images)
    real_loss = loss_function(real_output, real_label)
    real_loss.backward()

    # Generate fake images using the generator
    fake_images = make_random_images(actual_batch_size, generator)
    # label all fake images as 0.0
    fake_label = torch.full((actual_batch_size, 1), fake_label_val, device=device)

    # Get the derivative for the fake images
    fake_output = discriminator(fake_images.detach())
    fake_loss = loss_function(fake_output, fake_label)
    fake_loss.backward()

    # Discriminator total loss
    disc_loss = real_loss.item() + fake_loss.item()

    # Train the discriminator
    disc_opt.step()

    # Get derivative for the generator
    # We're adjusting the generator to make the
    # discriminator think fake images are real
    gen_opt.zero_grad()
    trick_output = discriminator(fake_images)
    trick_loss = loss_function(trick_output, real_label)
    trick_loss.backward()

    # Train the generator
    gen_opt.step()

    # Generator loss
    gen_loss = trick_loss.item()

    # Return discriminator loss and generator loss for logging
    return gen_loss, disc_loss
# task 15
def save_models(discriminator, generator, epoch, directory):
    # Get discriminator state dictionary
    disc_state_dict = discriminator.state_dict()

    disc_filename = directory / f"discriminator_{epoch}.pth"

    # Save discriminator save dictionary to `disc_filename`
    torch.save(disc_state_dict, disc_filename)
    # Get generator state dictionary
    gen_state_dict = generator.state_dict()

    gen_filename = directory / f"generator_{epoch}.pth"

    # Save generator save dictionary to `gen_filename`
    torch.save(gen_state_dict, gen_filename)

save_models(discriminator, generator, "untrained", now_dir)

print(f"Files in {now_dir}")
for filename in now_dir.glob("*"):
    print("\t", filename)

# task 16
def train_epoch(
    discriminator,
    generator,
    real_image_loader,
    loss_function,
    disc_opt,
    gen_opt,
    epoch,
    device=device,
):
    # train the model
    total_disc_loss = 0
    total_gen_loss = 0
    for real_data_batch in tqdm(real_image_loader):
        disc_loss, gen_loss = perform_batch_step(
            discriminator,
            generator,
            real_data_batch,
            loss_function,
            disc_opt,
            gen_opt,
            device,
        )
        # Keep a running total of losses from each batch
        total_disc_loss += disc_loss
        total_gen_loss += gen_loss

    # Save the models at the current epoch
    save_models(discriminator, generator, epoch, now_dir)
    print(f"Epoch {epoch} finished")
    print(f"Discriminator loss: {disc_loss}, Generator loss: {gen_loss}")

    # Create 6 images
    sample_images = make_random_images(6, generator)

    # Display the images
    display_images(sample_images)

total_epochs = 3

for epoch in range(1, total_epochs):
    train_epoch(
        discriminator,
        generator,
        dataloader,
        loss_function,
        disc_opt,
        gen_opt,
        epoch=epoch,
    )


# Load discriminator weights
disc_weights = torch.load("discriminator_99.pth")
discriminator.load_state_dict(disc_weights)

# Load generator weights
gen_weights = torch.load("generator_99.pth")
generator.load_state_dict(gen_weights)
# task 17
# Create 6 images
sample_images = make_random_images(6, generator)

# Display the images
display_images(sample_images)

single_batch = next(iter(dataloader))[0]
display_images(single_batch)