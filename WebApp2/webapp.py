%load_ext autoreload
%autoreload 2

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


# task 1

mtcnn =  MTCNN(image_size=240, keep_all =True, min_face_size=40)
resnet = InceptionResnetV1(pretrained="vggface2")

resnet = resnet.eval()

print(f"MTCNN image size: {mtcnn.image_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")
print(f"InceptionResnet weight set: {resnet.pretrained}")

# task 2
embedding_data = torch.load("embeddings.pt")

print(f"Known names: {[data[1] for data in embedding_data]}")

# task 3
project_dir = Path("project4")
images_dir = project_dir / "data"/ "extracted_frames"

print(images_dir)

# task 4
def locate_faces(image):
    cropped_images, probs = mtcnn(image, return_prob=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))

face = multiple_faces[0]
print(f"First bounding box: {face[0]}")
print(f"First probability: {face[1]}")
print(f"Shape of first cropped image: {face[2].shape}")
# task 5

def determine_name_dist(cropped_image, threshold=0.9):
    # Use `resnet` on `cropped_image` to get the embedding.
    # Don't forget to unsqueeze!
    emb = resnet(cropped_image.unsqueeze(0))

    # We'll compute the distance to each known embedding
    distances = []
    for known_emb, name in embedding_data:
        # Use torch.dist to compute the distance between
        # `emb` and the known embedding `known_emb`
        dist = torch.dist(emb, known_emb).item()
        distances.append((dist, name))

    # Find the name corresponding to the smallest distance
    dist, closest = min(distances)

    # If the distance is less than the threshold, set name to closest
    # otherwise set name to "Undetected"
    if dist < threshold:
        name = closest
    else:
        name = "undetected"

    return name, dist

print("Who's in the picture with 5 faces, with distances?")
for index, face in enumerate(multiple_faces):
    print(f"{index}: {determine_name_dist(face[2])}")

# task 6

def label_face(name, dist, box, axis):
    """Adds a box and a label to the axis from matplotlib
    - name and dist are combined to make a label
    - box is the four corners of the bounding box for the face
    - axis is the return from fig.subplots()
    Call this in the same cell as the figure is created"""

    # Add the code to generate a Rectangle for the bounding box
    # set the color to "blue" and fill to False
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    axis.add_patch(rect)

    # Set color to be red if the name is "Undetected"
    # otherwise set it to be blue
    if name == "undetected":
        color = "red"
    else:
        color = "blue"
    
    label = f"{name} {dist:.2f}"
    axis.text(box[0], box[1], label, fontsize="large", color=color)

# This sets the image size
# and draws the original image
width, height = sample_multiple.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(sample_multiple)
plt.axis("off")

face = multiple_faces[0]
cropped_image = face[2]
box = face[0]

name, dist = determine_name_dist(cropped_image)

label_face(name, dist, box, axis)

# task 7
    # This sets the image size
    # and draws the original image
width, height = sample_multiple.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(sample_multiple)
plt.axis("off")

for face in multiple_faces:
    box, prob, cropped_image = face

    name, dist = determine_name_dist(cropped_image)

    label_face(name, dist, box, axis)

# task 8
#            the complete function in one go
def add_labels_to_image(image):
    # This sets the image size
    # and draws the original image
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    # Use the function locate_faces to get the individual face info
    faces = locate_faces(image)

    for box, prob, cropped in faces:
        # If the probability is less than 0.90,
        # It's not a face, skip this run of the loop with continue
        if prob < 0.9:
            continue
        
        # Call determine_name_dist to get the name and distance
        name, dist = determine_name_dist(cropped)

        # Use label_face to draw the box and label on this face
        label_face(name,dist, box, axis)

    return fig

# task 9
labeled_single = add_labels_to_image(sample_single)

# task 10
import flaskApp
test_multiple = flaskApp.add_labels_to_image(sample_multiple)
# task 11

# the app.py

# task 12
gunicorn --bind 0.0.0.0:8000 app:app


# task 13


# task 14