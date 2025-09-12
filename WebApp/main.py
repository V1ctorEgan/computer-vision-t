# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
# task 1
mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)


resnet = InceptionResnetV1(pretrained="vggface2")


print(f"MTCNN image size: {mtcnn.image_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")
print(f"InceptionResnet weight set: {resnet.pretrained}")