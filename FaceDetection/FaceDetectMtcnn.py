import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.utils import make_grid

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")

# task 1: initialize the model
# get mmore info on the.txt with the same name as this file
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)


print(mtcnn)

# task 2
curr_work_dir = Path.cwd() 

print(curr_work_dir)

# task 3
extracted_frames_dir = curr_work_dir / "project4" / "data" / "extracted_frames"

print(extracted_frames_dir)

# task 4
sample_image_filename = "frame_320.jpg"
sample_image_path = extracted_frames_dir / sample_image_filename

sample_image = Image.open(sample_image_path)
sample_image

# task 5: face detection
boxes, probs = mtcnn.detect(sample_image)

print("boxes type:", type(boxes))
print("probs type:", type(probs))

# boxes.shape returns (3,4) meaning 3 faces, but four cordinates / bounding box data (x,y,h,n)

# task 6
number_of_detected_faces = len(boxes)

print(number_of_detected_faces)

# task 7
num_faces = len(probs[probs > 0.99])

print(num_faces)

# task 8: putting the bounding boxes over the images in the picture:
fig, ax = plt.subplots()
ax.imshow(sample_image)

for box in boxes:
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    ax.add_patch(rect)
plt.axis("off")

# task 9
boxes, probs, landmarks = mtcnn.detect(sample_image, landmarks=True)

print("boxes type:", type(boxes))
print("probs type:", type(probs))
print("landmarks type:", type(landmarks))

# task 10
print(landmarks.shape)  # output: (3,5,2)

# task 11: adding landmarks
fig, ax = plt.subplots()
ax.imshow(sample_image)

for box, landmark in zip(boxes, landmarks):
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    ax.add_patch(rect)
    for point in landmark:
        ax.plot(point[0], point[1], marker="o", color="red")
plt.axis("off")

# task 12
faces = mtcnn(sample_image)

print(faces.shape)

# task 13
Grid = make_grid(faces, nrow=3)

print(Grid.shape)

# task 14
images_dir = curr_work_dir / "project4" / "data" / "images"
images_dir.mkdir(exist_ok = True)

# task 15
mary_kom_dir = images_dir / "mary_kom"

# Now Create `mary_kom` directory
mary_kom_dir.mkdir(exist_ok=True)

# task 16
mary_kom_img_paths = [extracted_frames_dir/ i for i in mary_kom_imgs]


print("Number of images we'll use:", len(mary_kom_img_paths))

# task 17
for image_path in mary_kom_img_paths:
    shutil.copy(image_path, mary_kom_dir)

# task 18
ranveer_dir = images_dir / "ranveer"

# Now Create `ranveer` directory
ranveer_dir.mkdir(exist_ok=True)

ranveer_imgs = [
    "frame_10.jpg",
    "frame_40.jpg",
    "frame_270.jpg",
    "frame_365.jpg",
    "frame_425.jpg",
]

# task 19
ranveer_img_paths = [extracted_frames_dir/ i for i in ranveer_imgs]

print("Number of images we'll use:", len(ranveer_img_paths))

# task 20
for image_path in ranveer_img_paths:
    shutil.copy(image_path, ranveer_dir)

print("Number of files in ranveer directory:", len(list(ranveer_dir.iterdir())))

# task 21

