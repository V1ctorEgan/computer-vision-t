# 1
train_dir = os.path.join("potato_dataset","train")
classes = os.listdir(train_dir)

print(classes)
print(f"Number of classes: {len(classes)}")

# 2
height = 224
width = 224


class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((height,width)),
    transforms.ToTensor()
]
)

print(transform)

# 3
dataset = datasets.ImageFolder(root=train_dir, transform=transform)

print("Length of dataset:", len(dataset))

# 4
dataset = datasets.ImageFolder(root=train_dir, transform=transform)

print("Length of dataset:", len(dataset))

# 5
mean, std = get_mean_std(dataset_loader)

print(f"Mean: {mean}")
print(f"Standard deviation: {std}")

# 6
mean=[0.4938, 0.5135, 0.4351]
std = [0.1843, 0.1563, 0.2023]
transform_norm = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

print(transform_norm)

# 7
# Important! don't change this.
g = torch.Generator()
g.manual_seed(42)

norm_dataset = datasets.ImageFolder(root=train_dir, transform=transform_norm)

# Important, DON'T change the `generator=g` parameter
train_dataset, val_dataset = random_split(norm_dataset,[0.8,0.2], generator=g)

print("Length of dataset:", len(norm_dataset))
print("Training data set size:", len(train_dataset))
print("Validation data set size:", len(val_dataset))

#8

# Important! don't change this.
g = torch.Generator()
g.manual_seed(42)

batch_size = 32

# Important! Don't change the `generator=g` parameter
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True generator=g)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

print(type(train_loader))
print(type(val_loader))

# 9
# Important! Don't change this
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = torch.nn.Sequential()

conv1_n_kernels = 16
conv1 = torch.nn.Conv2d(
    in_channels=3, out_channels=conv1_n_kernels, kernel_size=(3, 3), padding=1
)
max_pool_size = 4
max_pool1 = torch.nn.MaxPool2d(max_pool_size)
model.append(conv1)
model.append(torch.nn.ReLU())
model.append(max_pool1)

conv2_n_kernels = 32
conv2 = torch.nn.Conv2d(
    in_channels=16, out_channels=conv2_n_kernels, kernel_size=(3, 3), padding=1
)
max_pool2 = torch.nn.MaxPool2d(max_pool_size)
model.append(conv2)
model.append(torch.nn.ReLU())
model.append(max_pool2)

conv3_n_kernels = 64
conv3 = torch.nn.Conv2d(32, conv3_n_kernels, 3, padding=1)
max_pool3 = torch.nn.MaxPool2d(max_pool_size)
model.append(conv3)
model.append(torch.nn.ReLU())
model.append(max_pool3)

model.append(torch.nn.Flatten())
model.append(torch.nn.Dropout())

linear1 = torch.nn.Linear(in_features=576, out_features=500)
model.append(linear1)
model.append(torch.nn.ReLU())
model.append(torch.nn.Dropout())

n_classes = 3
output_layer = torch.nn.Linear(500, n_classes)
model.append(output_layer)

summary(model, input_size=(batch_size, 3, height, width))

# 10
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Make sure to send the model to the GPU

print(loss_fn)
print("----------------------")
print(optimizer)
print("----------------------")
print(next(model.parameters()).device)

# 11
# Import the train function from `training.py`
from training import train 
# Train the model for 15 epochs
epochs = 15

train_losses, val_losses, train_accuracies, val_accuracies = train(model,optimizer,loss_fn,train_loader, val_loader, epochs, device=device, )
 # 12
 from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

print(model)

# 13
for params in model.parameters():
    params.requires_grad = False
print(model)

# 14
# Important! Don't change this
torch.manual_seed(42)
torch.cuda.manual_seed(42)

in_features = model.fc.in_features

modified_last_layer = nn.Sequential()

dense_layer = nn.Linear(in_features=in_features, out_features=256)
modified_last_layer.append(dense_layer)

relu = nn.ReLU()
modified_last_layer.append(relu)

modified_last_layer.append(nn.Dropout(p=0.5))

output_layer = nn.Linear(in_features=256, out_features = 3)
modified_last_layer.append(output_layer)

# Assign `modified_last_layer` to `model.fc`
model.fc = modified_last_layer

print(model)

# 15
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Place the model on device
model.to(device)

# Train the model for 10 epochs
epochs = 10
train_losses, val_losses, train_accuracies, val_accuracies = train(
    model, optimizer, loss_fn, train_loader, val_loader, epochs, device=device
)

# 16
from training import predict

test_dir = os.path.join("potato_dataset","test")

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_norm)
batch_size = 10

test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

print("Number of test images:", len(test_dataset))
# 17

# Predict the probabilities for each test image
test_probabilities = predict(model, test_loader, device)

# Get the index associated with the largest probability for each test image
test_predictions = torch.argmax(test_probabilities, dim=1)
# Converts the class index to the class name for each test image.
test_classes = [train_dataset.dataset.classes[i] for i in test_predictions]

print("Number of predictions:", test_predictions.shape)
print("Predictions (class index):", test_predictions.tolist())
print()
print("Predictions (class name):", test_classes)
#18
# Import the early_stopping function
from training import early_stopping
epochs_to_train = 50
checkpoint_path = "LR_model.pth"
early_stopping_function = early_stopping

train_results = train_callbacks(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=epochs_to_train,
    device=device,
    checkpoint_path=checkpoint_path,
    early_stopping=early_stopping_function,
)

(
    learning_rates,
    train_losses,
    valid_losses,
    train_accuracies,
    valid_accuracies,
    epochs,
) = train_results

#19
# Load the model with `torch.load`
checkpoint =torch.load("LR_model.pth")
# Load model state dictionaries
model.load_state_dict(checkpoint["model_state_dict"])

print(model)