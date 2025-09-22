import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1
from IPython.display import Video
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
data_dir = Path("data")

print(data_dir)

# task 2
video_name = "lupita_nyongo.mp4"

# task 3
input_video = data_dir / video_name

print(input_video)

# Display the video
Video(input_video, width=400)

# task 4
frames_dir = data_dir / "extracted_frames"

frames_dir.mkdir(exist_ok=True)
print(frames_dir)

#  let's create a video capture and compute the frame rate.
video_capture = cv2.VideoCapture(input_video)
frame_rate = round(video_capture.get(cv2.CAP_PROP_FPS))

print(f"Frame rate: {frame_rate}")
# task 5
# extract individual frames!
interval = 6  # Extract every sixth frame from the video
frame_count = 0

print("Start extracting individual frames...")
while True:
    # read the next frame from the video_capture
    ret, frame = video_capture.read()
    if not ret:
        print("Finished!")
        break  # Break the loop if there are no more frames

    # Save frames at every 'interval' frames
    if frame_count % interval  == 0:
        frame_path = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

    frame_count += 1

video_capture.release()

images_dir = data_dir / "images"
images_dir.mkdir(exist_ok=True)

print(images_dir)
# task 6
lupita_dir = images_dir / "lupita"
# Now create `lupita` directory
lupita_dir.mkdir(exist_ok=True)
christoph_dir =  images_dir / "christoph"
# Now create `christoph` directory
christoph_dir.mkdir(exist_ok=True)


lupita_imgs = [
    "frame_3438.jpg",
    "frame_3486.jpg",
    "frame_3852.jpg",
    "frame_4062.jpg",
    "frame_4914.jpg",
    "frame_4866.jpg",
]

christoph_imgs = [
    "frame_54.jpg",
    "frame_66.jpg",
    "frame_72.jpg",
    "frame_108.jpg",
    "frame_186.jpg",
    "frame_246.jpg",
]
# task 7
lupita_img_paths =[frames_dir/ img for img in lupita_imgs]
christoph_img_paths = [frames_dir/ img for img in christoph_imgs]

print("Number of Lupita images:", len(lupita_img_paths))
print("Number of Christoph images:", len(christoph_img_paths))

# load to see images
fig, axs = plt.subplots(1, 6, figsize=(10, 8))
for i, ax in enumerate(axs):
    ax.imshow(Image.open(lupita_img_paths[i]))
    ax.axis("off")

fig, axs = plt.subplots(1, 6, figsize=(10, 8))
for i, ax in enumerate(axs):
    ax.imshow(Image.open(christoph_img_paths[i]))
    ax.axis("off")

# task 8

# Copy selected images of Lupita over to `lupita` directory
for image_path in lupita_img_paths:
    shutil.copy(image_path, lupita_dir)

# Copy selected images of Christoph over to `christoph` directory
for image_path in christoph_img_paths:
    shutil.copy(image_path, christoph_dir)

print("Number of files in lupita directory:", len(list(lupita_dir.iterdir())))
print("Number of files in christoph directory:", len(list(christoph_dir.iterdir())))

# task 9
# face detection

mtcnn = MTCNN(device=device, keep_all=True, min_face_size=40)

print(f"MTCNN min face size: {mtcnn.min_face_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")

sample_image_filename = "frame_4866.jpg"
sample_image_path = frames_dir / sample_image_filename

sample_image = Image.open(sample_image_path)
sample_image

# task 10

boxes, probs, landmarks = mtcnn.detect(sample_image, landmarks=True)

print("boxes:", boxes)
print("probs:", probs)
print("landmarks:", landmarks)

# task 11
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

# face recognition
resnet = InceptionResnetV1(pretrained="vggface2").eval()

print(f"InceptionResnet weight set: {resnet.pretrained}")

# task 13

dataset = datasets.ImageFolder(images_dir)

print(dataset)


# task 14
idx_to_class = { d: i for i, d in dataset.class_to_idx.items()}

print(idx_to_class)

def collate_fn(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_fn)
print(loader.dataset)

# task 15
# Dictionary that maps name to list of their embeddings
name_to_embeddings = {name: [] for name in idx_to_class.values()}

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob >= 0.9:
        emb = resnet(face[0].unsqueeze(0))
        name_to_embeddings[idx_to_class[idx]].append(emb)

print(name_to_embeddings.keys())
print(type(name_to_embeddings["lupita"]))
print(type(name_to_embeddings["christoph"]))


# task 16

embeddings_lupita = torch.stack(name_to_embeddings["lupita"])
embeddings_christoph = torch.stack(name_to_embeddings["christoph"])

print(f"Shape of stack of embeddings for Lupita: {embeddings_lupita.shape}")
print(f"Shape of stack of embeddings for Christoph: {embeddings_christoph.shape}")

# task 17
avg_embedding_lupita = torch.mean(embeddings_lupita, dim=0)
avg_embedding_christoph = torch.mean(embeddings_christoph, dim=0)

print(f"Shape of avg_embedding_lupita: {avg_embedding_lupita.shape}")
print(f"Shape of avg_embedding_christoph: {avg_embedding_christoph.shape}")

test_images = ["frame_2658.jpg", "frame_4614.jpg", "frame_972.jpg", "frame_30.jpg"]
# task 18
# testing the model
test_paths = [frames_dir / frame for frame in test_images]

fig, axs = plt.subplots(1, len(test_paths), figsize=(10, 8))
for i, ax in enumerate(axs):
    ax.imshow(Image.open(test_paths[i]))
    ax.axis("off")

# task 19

from utils import recognize_faces

recognize_faces

embedding_list = [avg_embedding_lupita, avg_embedding_christoph]
name_list = ["lupita", "christoph"]

embedding_data = list(zip(embedding_list,name_list))

print(embedding_data[0][0].shape, embedding_data[0][1])
print(embedding_data[1][0].shape, embedding_data[1][1])
# task 20
recognized_faces = []
for test_img_path in test_paths:
    # Call recognize_faces function using test_img_path
    result = recognize_faces(test_img_path, embedding_data, mtcnn, resnet, threshold=0.9)
    # and append the result to the list `recognized_faces`
    recognized_faces.append(result)

