import pathlib
import sys

import matplotlib.pyplot as plt
import torch
import torchinfo
import torchvision
import ultralytics
from PIL import Image
from torchvision.transforms import v2
from ultralytics import YOLO

CLASS_DICT = dict(
    enumerate(
        [
            "ambulance",
            "army vehicle",
            "auto rickshaw",
            "bicycle",
            "bus",
            "car",
            "garbagevan",
            "human hauler",
            "minibus",
            "minivan",
            "motorbike",
            "pickup",
            "policecar",
            "rickshaw",
            "scooter",
            "suv",
            "taxi",
            "three wheelers (CNG)",
            "truck",
            "van",
            "wheelbarrow",
        ]
    )
)

print("CLASS_DICT type,", type(CLASS_DICT))
CLASS_DICT

runs_dir = pathlib.Path("runs", "detect")
list(runs_dir.glob("train*"))

# task 1
run_dir = runs_dir / "train"
weights_file =run_dir / "weights" /"best.pt"
print("Weights file exists?", weights_file.exists())

# task 2
model = YOLO(weights_file)

torchinfo.summary(model)

# task 3
result = model.train(
    data=model.overrides["data"],
    epochs=1,
    batch=8,
    workers=1,
)

# task 4
loader = model.trainer.train_loader

print(type(loader))

# task 5
for batch in loader:
    break
print(type(batch))

# task 6
print(batch.keys())

# task 7
print(batch['img'].shape)

# task 8
print(batch['bboxes'].shape)

print(batch["batch_idx"])

# task 9
def plot_with_bboxes(img, bboxes, cls, batch_idx=None, index=0, **kw):
    """Plot the bounding boxes on an image.

    Input:  img     The image, either as a 3-D tensor (one image) or a
                    4-D tensor (a stack of images).  In the latter case,
                    the index argument specifies which image to display.
            bboxes  The bounding boxes, as a N x 4 tensor, in normalized
                    XYWH format.
            cls     The class indices associated with the bounding boxes
                    as a N x 1 tensor.
            batch_idx   The index of each bounding box within the stack of
                        images.  Ignored if img is 3-D.
            index   The index of the image in the stack to be displayed.
                    Ignored if img is 3-D.
            **kw    All other keyword arguments are accepted and ignored.
                    This allows you to use dictionary unpacking with the
                    values produced by a YOLO DataLoader.
    """
    if img.ndim == 3:
        image = img[None, :]
        index = 0
        batch_idx = torch.zeros((len(cls),))
    elif img.ndim == 4:
        # Get around Black / Flake8 disagreement
        indp1 = index + 1
        image = img[index:indp1, :]

    inds = batch_idx == index
    res = ultralytics.utils.plotting.plot_images(
        images=image,
        batch_idx=batch_idx[inds] - index,
        cls=cls[inds].flatten(),
        bboxes=bboxes[inds],
        names=CLASS_DICT,
        threaded=False,
        save=False,
    )

    return Image.fromarray(res)

plot_with_bboxes(**batch, index=0)

# task 10
Image.open(batch('im_file')[0]) # but the actual path to the image 

yolo_base = pathlib.Path("data_yolo")
sample_fn = next((yolo_base / "images").glob("*/01.jpg"))
sample_labels = next((yolo_base / "labels").glob("*/01.txt"))

print(sample_fn)
print(sample_labels)

# task 11
sample_image = Image.open(sample_fn)

sample_image

# task 12
sample_torch =v2.ToImage()(sample_image)

print(sample_torch.shape)

!head -n5 $sample_labels

# task 13
# Load the data into `label_data`
with open(sample_labels, "r") as f:
    label_data = [row.split() for row in f]

label_data[:5]

# task 14
classes = torch.Tensor([[int(row[0])] for row in label_data ])

print("Tensor shape:", classes.shape)
print("First 5 elements:\n", classes[:5])

# task 15
bboxes = torch.Tensor([[float(el) for el in row[1:]] for row in label_data])


print("Tensor shape:", bboxes.shape)
print("First 5 elements:\n", bboxes[:5])

# task 16
sample_width, sample_height = sample_image.size

scale_factor = torch.Tensor([sample_width, sample_height, sample_width, sample_height])

bboxes_pixels = bboxes * scale_factor

print("Tensor shape:", bboxes_pixels.shape)
print("First 5 elements:\n", bboxes_pixels[:5])


bboxes_tv = torchvision.tv_tensors.BoundingBoxes(
    bboxes_pixels,
    format="CXCYWH",
    # Yes, that's right.  Despite using width x height everywhere
    # else, here we have to specify the image size as height x
    # width.
    canvas_size=(sample_height, sample_width),
)

print("Tensor type:", type(bboxes_tv))
print("First 5 elements:\n", bboxes_tv[:5])


# task 17
#  Use the RandomHorizontalFlip transformation to flip the sample image. Set p=1 to ensure that the flip happens.
flipped = v2.RandomHorizontalFlip(p=1)(sample_torch)

plot_with_bboxes(flipped, bboxes_tv, classes)

# task 18
flipped, flipped_bboxes = v2.RandomHorizontalFlip(p=1)(sample_torch, bboxes_tv)

plot_with_bboxes(flipped, flipped_bboxes, classes)

# task 19
rotated, rotated_bboxes = v2.RandomRotation(90)(sample_torch, bboxes_tv)

plot_with_bboxes(rotated, rotated_bboxes, classes)

# task 20
transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=90)
    ]
)

transformed, transformed_bboxes = transforms(sample_torch, bboxes_tv)

plot_with_bboxes(transformed, transformed_bboxes, classes)

# task 21
transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(640, 640)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    ]
)

transformed, transformed_bboxes = transforms(sample_torch, bboxes_tv)
plot_with_bboxes(transformed, transformed_bboxes, classes)