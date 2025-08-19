import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from IPython.display import Video
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, make_grid

# task 1
dhaka_image_dir = Path("data_images", "train")

print("Data directory:", dhaka_image_dir)

# task 2
images_dir = dhaka_image_dir / "images"
annotations_dir = dhaka_image_dir / "annotations"

images_dir.mkdir(exist_ok=True)
annotations_dir.mkdir(exist_ok=True)

dhaka_files = list(dhaka_image_dir.iterdir())
dhaka_files[-5:]
# task 3
for file in dhaka_files:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        target_dir = images_dir
    elif file.suffix.lower() == ".xml":
        target_dir = annotations_dir
    file.rename(target_dir / file.name)

#The ElementTree (ET) module in Python can parse an XML file. In XML, the root is the top-level element that 
# contains all other elements. The tag attribute contains the name of the element.
# the root is top-most level element(e.g <annotation> )
xml_filepath = annotations_dir / "01.xml"
tree = ET.parse(xml_filepath)
root = tree.getroot()
print(root.tag) # this output annotation

# task 4
bounding_boxes = []
labels = []
for obj in root.findall("object"):
    label = obj.find("name").text
    labels.append(label)
    bndbox = obj.find("bndbox")
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    xmax = bndbox.find("xmax").text
    ymax = bndbox.find("ymax").text
    bounding_boxes.append([xmin, ymin, xmax, ymax])

for label, bounding_box in zip(labels, bounding_boxes):
    print(f"{label}: {bounding_box}")

# task 5
bboxes_tensor = torch.tensor(bounding_boxes, dtype=torch.float)

print(bboxes_tensor)

# task 6
image_path = images_dir / "01.jpg"
image = read_image(str(image_path))
print(image.shape)

# task 7
image = draw_bounding_boxes(
    image=image,
    boxes=bboxes_tensor,
    labels=labels,
    width=3,
    fill=False,
    font="arial.ttf",
    font_size=10,
)

# task 8
video_dir = Path("data_video")
video_name = "dhaka_traffic.mp4"
video_path = video_dir / video_name

print(video_path)

# task 9

# """ now we turn the video into images then draw bounding boxes in each of the images"""
''' which is 9,333 images from a 6mins 33secs video'''
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {frame_rate}")
    print(f"Total number of frames: {frame_count:,}")

# checking the first image
success, first_frame = video_capture.read()
if success:
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
else:
    print("Error: Could not read frame.")


# checking the 100th image
video_capture.set(cv2.CAP_PROP_POS_FRAMES, 100)
success, later_frame = video_capture.read()
if success:
    plt.imshow(cv2.cvtColor(later_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
else:
    print("Error: Could not read frame.")

# now we create a directory for the images and save them
frames_dir = video_dir / "extracted_frames"
frames_dir.mkdir(exist_ok=True)

# now i walk through the video
frame_count = 0

while True:
    success, frame = video_capture.read()
    if not success:
        break

    # Save frames at the frame_rate
    if frame_count % frame_rate == 0:
        frame_path = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

    frame_count += 1

video_capture.release()

# now I'll display some images
def display_sample_images(dir_path, sample=5):
    image_list = []
    images = sorted(dir_path.iterdir())
    if images:
        sample_images = images[:sample]
        for sample_image in sample_images:
            image = read_image(str(sample_image))
            resize_transform = transforms.Resize((240, 240))
            image = resize_transform(image)
            image_list.append(image)
    grid = make_grid(image_list, nrow=5)
    image = to_pil_image(grid)
    return image


display_sample_images(frames_dir, sample=10)