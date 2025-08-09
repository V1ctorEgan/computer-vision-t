import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO

# task 1

if torch.cuda.is_available():
      device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device.")

# task 2
istanbul_dir = Path("istanbul_traffic", "train")

print("Training data directory:", istanbul_dir)

# task 3
file_extension_counts = Counter(p.stem for p in istanbul_dir.glob("*"))
file_extension_counts

# task 4
unique_counts =set(file_extension_counts.values())
unique_counts

# task 5
def parse_annotations(f):
    """Parse all of the objects in a given XML file to YOLO format.

    Input:  f      The path of the file to parse.

    Output: A list of objects in YOLO format.
            Each object is a list of numbers [class_id, x_center, y_center, width, height].
    """

    objects = []

    tree = ET.parse(f)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall("object"):
        class_id = int(obj.find("name").text)
        bndbox = obj.find("bndbox")
        # Getting the bounding box values
        x_c =float( bndbox.find("x_c").text)
        y_c = float(bndbox.find("y_c").text)
        width = float(bndbox.find("width").text)
        height = float(bndbox.find("height").text)

        # Appending the object in the form of
        # [class_id, x_center, y_center, width, height]
        objects.append([class_id, x_c, y_c, width, height])

    return objects
objects = parse_annotations(istanbul_dir / "0ab6f274892b9b370e6441886b2d7b9d.xml")
print(objects[0])


#task 6
def write_label(objects, filename):
    """Write the annotations to a file in the YOLO text format.

    Input:  objects   A list of YOLO objects, each a list of numbers.
            filename  The path to write the text file."""

    with open(filename, "w") as f:
        for obj in objects:
            # Write the object out as space-separated values
            f.write(" ".join(str(x) for x in obj))
            # Write a newline
            f.write("\n")


# task 7
yolo_base = Path("data_yolo")
# Make sure everything's cleared out
shutil.rmtree(yolo_base, ignore_errors=True)

(yolo_base / "images" / "train").mkdir(parents=True)
# Create the remaining directories.
(yolo_base / "images" / "val").mkdir(parents=True)
(yolo_base / "labels" / "train").mkdir(parents=True)
(yolo_base / "labels" / "val").mkdir(parents=True)

# task 8
# Don't change this
random.seed(42)

train_frac = 0.8
images = list(istanbul_dir.glob("*.jpg"))

for img in tqdm(images):
    # Randomly choose train or val split
    split = "train" if random.random() < train_frac else "val"
    # this should be `train` or `val`
    # XML file path, from image stem
    annotation = istanbul_dir / f"{img.stem}.xml"
    # Parse annotations.  Watch out for errors!
    try:
        parsed = parse_annotations(annotation)
    except Exception as e:
        print(f"failed to parse {img.stem} Skipping")
        print(e)
        continue
    
    # Write label file based on parsed annotation
    dest = yolo_base / "labels" / split / f"{img.stem}.txt"
    write_label(parsed, dest)

    dest = yolo_base / "images" / split / f"{img.stem}.jpg"
   
    
    # Copy image file to correct location
    shutil.copy(img, dest)

# task 9
metadata = {
    "path": str(
        yolo_base.absolute()
    ),  # It's easier to specify absolute paths with YOLO.
    
    "train": "images/train", # Training images, relative to above.
    
    "val": "images/val", # Validation images
    
    "names": classes, # Class names, as a list
    
    "nc": len(classes), # Number of classes
}

print(metadata)

# task 10
yolo_config = "data.yaml"
yaml.safe_dump(metadata, open(yolo_config, 'w'))

!cat data.yaml

# task 11
saved_model = YOLO("runs/detect/train/weights/best.pt")

# task 12
save_dir = Path("runs", "detect", "train") 
save_dir

# task 13
pr_curve_image = Image.open(save_dir/ "PR_curve.png")
pr_curve_image

# task 14

image_path_task = Path(
    "istanbul_traffic", "test", "3c794894a576d0d6355379613c2dadc5.jpg"
)

result = saved_model.predict(image_path_task, conf=0.5, save=True)

print(type(result))
# task 15

num_detections = len(result[0].boxes.cls)
print(f"Number of objects detected: {num_detections }")

# task 16
detected_objects = Counter(
    [saved_model.names[int(idx)] for idx in result[0].boxes.cls]
)  
print(detected_objects)

# task 17
total_time = sum(result[0].speed.values())

print(f"Total time in milliseconds: {total_time}")

# task 18
location_of_results = Path(result[0].save_dir)
print(f"Location of saved results: {location_of_results}")

# task 19
test_images_path = Path("istanbul_traffic", "test")
results_test = saved_model.predict(test_images_path, conf=0.5, save=True)

# task 20
detected_objects_test = Counter(
    [saved_model.names[int(idx)] for result in results_test for idx in result.boxes.cls]
)