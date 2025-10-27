# Import the libraries that you need
from pathlib import Path

import matplotlib.pyplot as plt
import medigan
import torch
import torch.optim as optim
import torchvision
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

# no 3
# Create the connection to the Medigan generators
generators = medigan.Generators()

# Find the models that match what we want
values =["mammogram", "roi"]
models = generators.find_matching_models_by_values(values=values, is_case_sensitive=False)
model_id = models[0].model_id

print(model_id)

# no 4
model_config = generators.get_config_by_id(model_id= model_id)

print(f"Model keys: {model_config.keys()}")

# no 5

model_info =  {
    k: model_config["selection"][k] for k in ["generates", "tags", "height","width"]
}

model_info

# no 6
output_dir = Path("output")
sample_dir = output_dir / "sample_mammogram"

# Create the directory with mkdir
sample_dir.mkdir(parents=True, exist_ok=True)
print(sample_dir)

# no 7
generators.generate(
    model_id=model_id,
    num_samples=4,
    output_path=sample_dir,
)


# no 8
def view_images(directory, num_images=4, glob_rule="*.jpg"):
    """Displays a sample of images in the given directory
    They'll display in rows of 4 images
    - directory: which directory to look for images
    - num_images: how many images to display (default 4, for one row)
    - glob_rule: argument to glob to filter images (default "*" selects all)"""

    image_list = list(directory.glob(glob_rule))
    num_samples = min(num_images, len(image_list))
    images = [read_image(str(f)) for f in sorted(image_list)[:num_samples]]
    grid = make_grid(images, nrow=4, pad_value=255.0)
    return torchvision.transforms.ToPILImage()(grid)


sample_images = view_images(sample_dir)
sample_images

# no 9
train_dataloader =  generators.get_as_torch_dataloader(
    model_id=model_id, num_samples=50, batch_size=4, shuffle=True, prefetch_factor=None
)

sample_batch = next(iter(train_dataloader))
print(f"Training data loader with keys: {sample_batch.keys()}")

# no 10
val_dataloader =  generators.get_as_torch_dataloader(
    model_id=model_id, num_samples=30, batch_size=4, shuffle=False, prefetch_factor=None
)

val_batch = next(iter(train_dataloader))
shape = val_batch["sample"].shape
dtype = val_batch["sample"].dtype
print(f"Validation image with data shape {shape} and type {dtype}")

# no 11
shape = val_batch["mask"].shape
dtype = val_batch["mask"].dtype

print(f"Validation mask with data shape {shape} and type {dtype}")

#We'll need to fix this, we need the images to be [3, 256, 256] and the mask to be [1, 256, 256], and 
# both to have type float32. This function converts the type and adds the channels.
def convert_to_torch_image(tensor, color=False):
    tensor_float = tensor.type(torch.float32)
    grayscale = tensor_float.unsqueeze(1)
    if color:
        return grayscale.repeat(1, 3, 1, 1)
    else:
        return grayscale
    
# no 12
mask_converted =  convert_to_torch_image(val_batch["mask"])

shape = mask_converted.shape
dtype = mask_converted.dtype

print(f"Validation mask with data shape {shape} and type {dtype}")

# no 13
sample_converted =  convert_to_torch_image(val_batch["sample"])

shape = sample_converted.shape
dtype = sample_converted.dtype

print(f"Validation mask with data shape {shape} and type {dtype}")

# no 14
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Load pretrained DeepLabV3 model with COCO + VOC labels
pretrained_weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=pretrained_weights)

print("Model components:")
for name, part in model.named_children():
    print("\t" + name)

# no 15
sample_converted = convert_to_torch_image(sample_batch["sample"], color=True).to(device)
# Put the model in evaluation mode and move it to the device
model = model.to(device)
model.eval()
with torch.no_grad():
    model_result = model(sample_converted)
model_out = model_result["out"]
out_shape = model_out.shape

out_shape

# no 16
# Replace the last layer in the classifier
new_final_layer = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
model.classifier[-1] = new_final_layer

# Move the entire model to the same device as your input
model = model.to(device)

# Run inference
with torch.no_grad():
    new_out = model(sample_converted)["out"]

print(f"New model output shape: {new_out.shape}")
print(f"Mask shape: {mask_converted.shape}")

# no 17
loss_fun = torch.nn.BCEWithLogitsLoss()

# Define the optimizer (Adam) — optimizing all model parameters
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

opt

# no 18
def compute_loss(batch, model, loss_fun):
    # Extract the sample and mask from the batch
    sample = batch["sample"]
    mask = batch["mask"]

    # Convert the sample and mask to the correct shape and type
    sample_correct = convert_to_torch_image(sample, color=True)
    mask_correct = convert_to_torch_image(mask)

    # move the sample and mask to the GPU (or CPU/MPS depending on device)
    sample_gpu = sample_correct.to(device)
    mask_gpu = mask_correct.to(device)

    # Run the model on the sample and select the classifier (out key)
    output = model(sample_gpu)["out"]

    # Compute the loss
    loss = loss_fun(output, mask_gpu)

    return loss

model.to(device)

compute_loss(sample_batch, model, loss_fun)

# no 19
def train_epoch(model, train_dataloader, val_dataloader, loss_fun, opt):
    model.train()

    # Training part
    train_loss = 0.0
    train_count = 0
    for batch in tqdm(train_dataloader):
        # zero the gradients on the optimizer
        opt.zero_grad()

        # compute the loss for the batch
        loss = compute_loss(batch, model, loss_fun)

        # Compute the backward part of the loss and step the optimizer
        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_count += 1

    # Validation part
    val_loss = 0.0
    val_count = 0
    for batch in tqdm(val_dataloader):
        # compute the loss for each batch
        loss = compute_loss(batch, model, loss_fun)

        val_loss += loss.item()
        val_count += 1

    return train_loss / train_count, val_loss / val_count

train_epoch(model, train_dataloader, val_dataloader, loss_fun, opt)

# no 20 
model = torch.load('model_trained.pth').to(device)

# no 21
test_dataloader =  generators.get_as_torch_dataloader(
    model_id=model_id, num_samples=8, batch_size=4, shuffle=False, prefetch_factor=None
)

test_batch = next(iter(test_dataloader))

print(f"Data loader images in batches of {test_batch['sample'].size(0)}")

# no 22
corrected_sample = convert_to_torch_image(test_batch["sample"], color=True)
corrected_sample = corrected_sample.to(device)

test_result = model(corrected_sample)["out"]

test_result.shape

# no 23
test_mask_model = torch.sigmoid(test_result)

def plot_images_from_tensor(tensor):
    grid = make_grid(tensor, nrow=4, pad_value=1.0)
    return torchvision.transforms.ToPILImage()(grid)

# no 24
# Plot the sample part of the test_batch
sample_test_batch_plot = plot_images_from_tensor(convert_to_torch_image(test_batch["sample"]))
sample_test_batch_plot

# Plot the mask part of the test_batch
mask_test_batch_plot = plot_images_from_tensor(convert_to_torch_image(test_batch["mask"]))
mask_test_batch_plot

# Plot the result of the model running
model_result_plot = plot_images_from_tensor(test_mask_model)
model_result_plot
