# **AI Lab: Deep Learning for Computer Vision**
# **WorldQuant University**
#
#

# **Usage Guidelines**
#
# This file is licensed under Creative Commons Attribution-NonCommercial-
# NoDerivatives 4.0 International.
#
# You **can** :
#
#   * ✓ Download this file
#   * ✓ Post this file in public repositories
#
# You **must always** :
#
#   * ✓ Give credit to WorldQuant University for the creation of this file
#   * ✓ Provide a link to the license
#
# You **cannot** :
#
#   * ✗ Create derivatives or adaptations of this file
#   * ✗ Use this file for commercial purposes
#
# Failure to follow these guidelines is a violation of your terms of service and
# could lead to your expulsion from WorldQuant University and the revocation
# your certificate.
#
#

# Import needed libraries
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
# Load MTCNN, Resnet, and the embedding data

mtcnn = MTCNN(image_size=240, keep_all =True, min_face_size=40)

resnet = InceptionResnetV1(pretrained="vggface2")
embedding_data = torch.load("embeddings.pt")

resnet = resnet.eval()


# Fill in the locate_face function
def locate_faces(image):
    cropped_images, probs = mtcnn(image, return_prob=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))
    


# Fill in the determine_name_dist function
def determine_name_dist(cropped_image, threshold=0.9):
    emb = resnet(cropped_image.unsqueeze(0))

    distances = []
    for known_emb, name in embedding_data:
     
        dist = torch.dist(emb, known_emb).item()
        distances.append((dist, name))

    dist, closest = min(distances)

    if dist < threshold:
        name = closest
    else:
        name = "undetected"

    return name, dist
    


# Fill in the label_face function
def label_face(name, dist, box, axis):
    """Adds a box and a label to the axis from matplotlib
    - name and dist are combined to make a label
    - box is the four corners of the bounding box for the       face
    - axis is the return from fig.subplots()
    Call this in the same cell as the figure is created"""
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],     fill=False, color="blue"
    )
    axis.add_patch(rect)

    # Set color to be red if the name is "Undetected"
    # otherwise set it to be blue
    if name == "undetected":
        color = "red"
    else:
        color = "blue"
    
    label = f"{name} {dist:.2f}"
    axis.text(box[0], box[1], label, fontsize="large",          color=color)

    


# Fill in the add_labels_to_image function
def add_labels_to_image(image):
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi),       dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

   
    faces = locate_faces(image)

    for box, prob, cropped in faces:
        
        if prob < 0.9:
            continue
        
        name, dist = determine_name_dist(cropped)

        label_face(name,dist, box, axis)

    return fig


# This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.
