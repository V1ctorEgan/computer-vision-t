import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")

# task 1
mtcnn0 = MTCNN(image_size=240, keep_all =False, min_face_size=40) # detect only one face
print(mtcnn0)

# task 2
#  initialize the Inception-ResNet V1 model.
resnet = InceptionResnetV1(pretrained="vggface2").eval()

images_folder = Path("project4", "data", "images")

print(f"Path to images: {images_folder}")

# task 3
dataset = datasets.ImageFolder(images_folder)

print(dataset)

# task 4
for subdirectory in images_folder.iterdir():
    print(subdirectory)

dataset.class_to_idx

# task 5
idx_to_class = {i : c for c, i in dataset.class_to_idx.items()}

print(idx_to_class)


def collate_fn(x):
    return x[0]

# task 6
loader = DataLoader(dataset, collate_fn=collate_fn)

print(loader.dataset)

img, _ = iter(loader).__next__()
img
# task 7
face, prob = mtcnn0(img, return_prob=True)

print(type(face))
print(f"Probability of detected face: {prob}")

try:
    resnet(face)
except ValueError as e:
    print(e)

# task 8
print(face.shape)

# task 9
face_4d = face.unsqueeze(0)

print(face_4d.shape)

embedding = resnet(face_4d)

print(f"Shape of face embedding: {embedding.shape}")

# task 10
# Dictionary that maps name to list of their embeddings
name_to_embeddings = {name: [] for name in idx_to_class.values()}

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob >= 0.9:
        emb = resnet(face.unsqueeze(0))
        name_to_embeddings[idx_to_class[idx]].append(emb)

print(name_to_embeddings.keys())
print(type(name_to_embeddings["mary_kom"]))

# task 11
embeddings_mary = torch.stack(name_to_embeddings["mary_kom"])
embeddings_ranveer = torch.stack(name_to_embeddings["ranveer"])

embeddings_mary_shape = embeddings_mary.shape
embeddings_ranveer_shape = embeddings_ranveer.shape

print(f"Shape of stack of embeddings for Mary: {embeddings_mary_shape}")
print(f"Shape of stack of embeddings for Ranveer: {embeddings_ranveer_shape}")

# task 12
avg_embedding_mary = torch.mean(embeddings_mary, dim=0)
avg_embedding_ranveer = torch.mean(embeddings_ranveer, dim=0)

print(f"Shape of avg_embedding: {avg_embedding_mary.shape}")
print(f"Shape of avg_embedding: {avg_embedding_ranveer.shape}")

# task 13
embeddings_to_save = [
    ( avg_embedding_mary, "mary_kom"),
    (avg_embedding_ranveer, "ranveer")
]

torch.save(embeddings_to_save, "embeddings.pt")

# task 14
embedding_data =torch.load("embeddings.pt")

names = [name for _, name in embedding_data]
print(f"Loaded the embedding for: {names}")

# task 15
test_img_path = Path("project4", "data", "extracted_frames", "frame_100.jpg")

test_img = Image.open(test_img_path)
test_img


mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)
print(f"MTCNN image size: {mtcnn.image_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")

# task 16
img_cropped_list, prob_list = mtcnn(test_img, return_prob=True)

print(f"Number of detected faces: {len(prob_list)}")
print(f"Probability of detected face: {prob_list[0]}")

for i, prob in enumerate(prob_list):
    if prob > 0.90:
        emb = resnet(img_cropped_list[i].unsqueeze(0))
# task 17

distances = {}

for known_emb, name in embedding_data:
    dist = torch.dist(emb, known_emb).item()
    distances[name] = dist

closest, min_dist = min(distances.items(), key=lambda x: x[1])
print(f"Closest match: {closest}")
print(f"Calculated distance: {min_dist :.2f}")


boxes, _ = mtcnn.detect(test_img)
print(f"Shape of boxes tensor: {boxes.shape}")

# task 18
# This sets the image size and draws the original image
width, height = test_img.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(test_img)
plt.axis("off")

for box in boxes:
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    axis.add_patch(rect)

    closest, min_dist = min(distances.items(), key=lambda x: x[1])

    # Drawing the box with recognition results

    if min_dist < threshold:
        name = closest
        color = "blue"
    else:
        name = "Unrecognized"
        color = "red"

    plt.text(
        box[0],
        box[1],
        f"{name} {min_dist:.2f}",
        fontsize=12,
        color=color,
        ha="left",
        va="bottom",
    )

plt.axis("off")
plt.show()

# task 19
img_multiple_people_path = Path("project4", "data", "extracted_frames", "frame_210.jpg")
img_multiple_people = Image.open(img_multiple_people_path)

img_multiple_people

# task 20
recognize_faces(img_multiple_people_path, embedding_data, mtcnn, resnet, threshold=0.9)
