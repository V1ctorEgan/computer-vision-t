
# Objectives:

# Detect objects in an image using the YOLO model
# Parse the results from running the YOLO model
# Display the resulting bounding boxes from running the YOLO model
# Detect objects from a variety of sources, including videos and images stored in directories
# Detect objects from a video source

import sys
from collections import Counter
from pathlib import Path

import PIL
import torch
import torchvision
import ultralytics
from IPython.display import Video
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from ultralytics import YOLO

yolo = YOLO(task="detect", model="yolov8s.pt")
# pt stands for pretrained

yolo.names
# this tells us what classes or object this model can detect

# task 1
class_assigned_to_23 = yolo.names[23]
print(f"{class_assigned_to_23} corresponds to 23")

# task 2
    # what of things that are not among the classes
classes_not_in_yolo = [
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers (CNG)",
    "van",
    "wheelbarrow",
]
"ambulance" not in yolo.names.values()


is_army_vehicle_inlcuded = "army vehicle" not in yolo.names.values()
print(is_army_vehicle_inlcuded)

data_dir = Path("data_video", "extracted_frames")
image_path = data_dir / "frame_1050.jpg"

result = yolo(image_path)

# task 3
image_path_task = data_dir / "frame_2575.jpg"
result_task = yolo(image_path_task)

print(type(result_task))

result = yolo.predict(image_path, conf=0.5, save=True, save_txt=True)
# .predict give better control, you can set the confidence level and save as txt file the result

# task 4
result_task =yolo.predict(image_path_task, conf=0.5, save=True, save_txt = True)

# task 5
# determine the length (nember) of the image it detected note! cls is short for classes
number_of_detected_objs =len(result_task[0].boxes.cls)
print(f"Number of objects detected in frame_2575.jpg: {number_of_detected_objs}")

object_counts = Counter([yolo.names[int(cls)] for cls in result[0].boxes.cls])
object_counts

# task 6
object_counts_task =Counter([yolo.names[int(cls)] for cls in result_task[0].boxes.cls])

most_common_class, count_of_class = object_counts_task.most_common(n=1)[0]
print(f"Most common class: {most_common_class}")
print(f"Number of detected {most_common_class}: {count_of_class}")

print(result[0].boxes.conf)
print(f"Number of objects detected: {len(result[0].boxes.conf)}")

# task 7
length_of_confidence_tensor = len(result_task[0].boxes.conf)
print(f"Number of objects detected: {length_of_confidence_tensor}")

number_of_confident_objects = (result[0].boxes.conf > 0.75).sum().item()
print(f"Number of objects detected with 50% confidence: {number_of_confident_objects}")

# task 8
number_of_confident_objects_task = (result_task[0].boxes.conf > 0.75).sum().item()

print(
    f"Number of objects detected in frame_2575.jpg with 50% confidence: {number_of_confident_objects_task}"
)

# task 9
original_shape_task = result_task[0].orig_shape
print(f"Original shape of frame_2574.jpg: {original_shape_task}")

# task 10
normalized_xywh = result_task[0].boxes.xywhn
print(f"Normalized xywh bounding box for frame_2575.jpg: {normalized_xywh[0]}")

# task 11
normalized_xywh_task = result_task[0].boxes.xywh / torch.Tensor([640, 360, 640, 360]).to("cuda")
print(f"Normalized xywh bounding box for frame_2575.jpg: {normalized_xywh[0]}")

# task 12
location_of_results_task = Path(result_task[0].save_dir)
print(f"Results for frame_2575.jpg saved to {location_of_results_task}")

# task 13
total_time = sum(result_task[0].speed.values())
print(f"Total time in milliseconds: {total_time}")

# task 14
# Display image frame_2575.jpg with the bounding boxes
Image.open(location_of_results / "frame_2575.jpg")

# task 15
with (location_of_results / "labels" / "frame_2575.txt").open("r") as f:
    print(f.read())

#now it is time to run yolo on multiple images
def display_sample_images(dir_path, sample=5):
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path

    image_list = []
    # Sort the images to ensure they are processed in order
    images = sorted(dir_path.glob("*.jpg"))
    if not images:
        return None

    # Iterate over the first 'sample' images
    for img_path in images[:sample]:
        img = read_image(str(img_path))
        resize_transform = transforms.Resize((240, 240))
        img = resize_transform(img)
        image_list.append(img)

    # Organize the grid to have 'sample' images per row
    Grid = make_grid(image_list, nrow=5)
    # Convert the tensor grid to a PIL Image for display
    img = torchvision.transforms.ToPILImage()(Grid)
    return img

# task 16
# Display the first ten images
display_sample_images(data_dir, sample=10)

# task 17
images_path_task = list(data_dir.iterdir())[-10:]

print(f"Number of frames in list: {len(images_path_task)}")
images_path_task

# task 18
results_task = yolo.predict(
    images_path_task,
    conf=0.5,
    save=True,
    save_txt=True,
    project=Path("runs", "detect"),
    name="multiple_frames_task",
)

print(f"\nResults from task saved to: {results_task[0].save_dir}")

# task 19
image_task = display_sample_images(results_task[0].save_dir, sample=10)
image_task

# we can give yolo a video and it will break it into multiple images and save to a directory for processing (like i just did)
video_path = Path("data_video", "dhaka_traffic.mp4")
Video(video_path)

# To speed things up, we're going to truncate our video and run YOLO against the truncated version. We'll use 
# ffmpeg, a command line tool for video and audio editing. The part that controls the timestamps for truncation 
# are the numbers that follow -ss and -to. The number after -ss is the starting timestamp and -to is the ending 
# timestamp. The value data_video/dhaka_traffic_truncated.mp4 is the path of the created file.

# !ffmpeg -ss 00:00:00 -to 00:00:30 -y -i $video_path -c copy data_video/dhaka_traffic_truncated.mp4
# !ffmpeg -ss 00:00:30 -to 00:01:00 -y -i $video_path -c copy data_video/dhaka_traffic_truncated_task.mp4

video_truncated_path_task = Path("data_video", "dhaka_traffic_truncated_task.mp4")
Video(video_truncated_path_task)

results_video = yolo.predict(
    video_truncated_path,
    conf=0.5,
    save=True,
    stream=True,
    project=Path("runs", "detect"),
    name="video_source",
)

for result in results_video:
    continue

# Using YOLO in the Command Line not python
# !yolo task=detect mode=predict conf=0.5 model=yolov8s.pt source=$video_truncated_path project="runs/detect" name="command_line" > /dev/null

