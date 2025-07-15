# task 1
dhaka_image_dir = Path("data_images", "train")

print("Data directory:", dhaka_image_dir)

# task 2
images_dir = dhaka_image_dir / "images"
annotations_dir = dhaka_image_dir / "annotations"

images_dir.mkdir(exist_ok=True)
annotations_dir.mkdir(exist_ok=True)

# task 3
for file in dhaka_files:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        target_dir = image_dir
    elif file.suffix.lower() == ".xml":
        target_dir = annotations_dir
    file.rename(target_dir / file.name)

