#task 1
data_dir = os.path.join("data_p2","data_undersampled","train")

print("Data directory:", data_dir)

#task 2
# mean = [0.4326, 0.4952, 0.3120]
# std = [0.2179, 0.2214, 0.2091]

mean = [0.4326, 0.4953, 0.3120]

std = [0.2178, 0.2214, 0.2091]

transform_normalized = transforms.Compose(
   [ConvertToRGB(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
       ]
)
transform_normalized

# task 7
# Move the model to device
model.to(device)

# Move our test_batch to device
test_batch_cuda = test_batch.to(device)

print("Test batch is running on:", test_batch_cuda.device)

# task 3
dataset = datasets.ImageFolder(root = data_dir, transform=transform_normalized)

dataset

# task 4
counts = class_counts(dataset)
counts

# task 5
batch_size = 32
dataset_loader = DataLoader(dataset, batch_size=batch_size)

print(f"Batch shape: {next(iter(dataset_loader))[0].shape}")

# task 6
test_batch = next(iter(dataset_loader))[0]
batch_shape = test_batch.shape

# Create the model summary
summary(model, input_size=batch_shape)

# task 7
# Move the model to device
model.to(device)

# Move our test_batch to device
test_batch_cuda = test_batch.to(device)

print("Test batch is running on:", test_batch_cuda.device)

# task 8
model_test_out = model(test_batch_cuda)
model_test_shape = model_test_out.shape

print("Output shape:", model_test_shape)

# task 9
in_features = model.fc.in_features
in_features

# task 10
classification_layer = torch.nn.Linear(in_features=in_features, out_features=256)


# Add the layer to our classifier
classifier.append(classification_layer)

# task 11
output_layer = torch.nn.Linear(in_features=256, out_features=5)

# Add the layer to our classifier
classifier.append(output_layer)


# task 12
# Create the model summary
summary(model, input_size=batch_shape)
# task 13 - k-fold
k = 5

kfold_splitter = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

train_nums, val_nums = next(kfold_splitter.split(range(100)))
fold_fraction = len(val_nums) / (len(train_nums) + len(val_nums))
print(f"One fold is {100*fold_fraction:.2f}%")

# task 14
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# task 15
# we'll use the following function to reset the model to avoid ofverfitting when doing the cross validation
def reset_classifier(model):
    model.fc.get_submodule("0").reset_parameters()
    model.fc.get_submodule("3").reset_parameters()

# you can safely skip this cell and load the model in the next cell

training_records = {}
fold_count = 0

for train_idx, val_idx in kfold_splitter.split(np.arange(len(dataset))):
    fold_count += 1
    print("*****Fold {}*****".format(fold_count))

    # Make train and validation data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Reset the model
    reset_classifier(model)

    # Train
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs=num_epochs,
        device=device,
        use_train_accuracy=False,
    )

    # Save training results for graphing
    training_records[fold_count] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    print("\n\n")

# task 16
# Plot the validation losses
plot_all_folds(training_records, "val_losses")

# task 17
# Plot the validation accuracies
plot_all_folds(training_records, "val_accuracies")

# task 18
probabilities = predict(model, val_loader, device)
predictions = torch.argmax(probabilities, dim=1)

# task 19
cm = confusion_matrix(targets, predictions.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.show();

