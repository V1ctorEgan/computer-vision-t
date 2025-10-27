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

generators = medigan.Generators()

generators.list_models()

generators = medigan.Generators()

generators.list_models()

model_00001_config = generators.get_config_by_id(model_id=1)
model_00001_config
# task 1
model_00001_info = {
    k: model_00001_config["selection"][k] for k in ["organ", "modality", "tags"]
}
model_00001_info


# task 2
model_00001_info['title'] = model_00001_config["description"]["title"]
model_00001_info['commemt'] = model_00001_config["description"]["comment"]

model_00001_info

# task 3
model_00007_config = generators.get_config_by_id(model_id = 7)

model_00007_info = {key: model_00007_config['selection'][key] for key in ["organ", "modality", "tags"]}
model_00007_info.update({key: model_00007_config['description'][key] for key in ["title", "comment"]})

model_00007_info

# task 4
key = "tags"
value = "mammogram"

found_models = generators.get_models_by_key_value_pair(
    key1=key, value1=value, is_case_sensitive=False
)

print(found_models)
print()
print(f"Found {len(found_models)} models")

model_id = found_models[1]["model_id"]

model_config = generators.get_config_by_id(model_id=model_id)

print("Model's description:")
model_config["description"]

# task 5
model_id = found_models[2]["model_id"]

model_config = generators.get_config_by_id(model_id=model_id)

print("Model's description:")
model_config["description"]

# task 6
values_list = ["lung", "xray", "1024"]

found_models = generators.find_matching_models_by_values(
    values=values_list,
    is_case_sensitive=False,
)

lung_xray_id = found_models[0].model_id

print(f"Lung x-ray model ID: {lung_xray_id}")

# task 7
output_dir = Path("output")
sample_dir = output_dir / "sample_lug"

print(sample_dir)

generators.generate(
    model_id=lung_xray_id,
    num_samples=8,
    output_path=sample_dir,
)

def view_images(directory, num_images=4, glob_rule="*"):
    """Displays a sample of images in the given directory
    They will display in rows of 4 images
    - directory: which directory to look for images
    - num_images: how many images to display (default 4, for one row)
    - glob_rule: argument to glob to filter images (default "*" selects all)"""

    image_list = list(directory.glob(glob_rule))
    num_samples = min(num_images, len(image_list))
    images = [read_image(str(f)) for f in sorted(image_list)[:num_samples]]
    grid = make_grid(images, nrow=4, pad_value=255.0)
    return torchvision.transforms.ToPILImage()(grid)

view_images(sample_dir)
# task 8
model_id = 10
polyp_dir = output_dir / "polyp_samples"

generators.generate(
    model_id=model_id,
    num_samples=8,
    output_path=polyp_dir,
)

print(
    f"Created {len(list(polyp_dir.glob('*')))} images (should be twice the number asked for)"
)

view_images(polyp_dir, 8, "*img*")

view_images(polyp_dir, 8, "*mask*")

# we can train it by making it return dataloader instead of files

train_dataloader = generators.get_as_torch_dataloader(
    model_id=10, num_samples=200, batch_size=4, shuffle=True, prefetch_factor=None
)