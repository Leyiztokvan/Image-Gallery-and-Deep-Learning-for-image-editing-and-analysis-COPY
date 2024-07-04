# Imports
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale

import CNN_utils as utils # Self implemented functions and classes used for this file

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("We are using {}".format(device))


##### Read data (should be extracted to correct directory by running Img_Gallery_Examples.py)
#
# Unzip file first in the same directory as this file to ensure path correctness or adapt directories
train_dir = "xray_dataset/chest_xray_64/train"
val_dir = "xray_dataset/chest_xray_64/val"

# Create train and val ImageFolder objects with images
train_dataset_ = ImageFolder(root=train_dir, is_valid_file=lambda x: x.endswith("jpeg"))
val_dataset_ = ImageFolder(root=val_dir, is_valid_file=lambda x: x.endswith("jpeg"))

# Display some images when still PIL images
utils.plot_CT_scans(dataset=train_dataset_, n_images=6, label=0, gray=True)
utils.plot_CT_scans(dataset=train_dataset_, n_images=6, label=1, gray=True)


# Setup composition of transforms
target_size = 64
nChannel = 1 # Greyscale

transforms = Compose([
                    Grayscale(1), # Use only as grayscale for efficiency
                    ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                    Normalize(mean=(0.5), std=(0.5)), # Normalize (not to mean 0 to avoid division by zero)
                    ])

# Wrap datasets with transformations
train_dataset = utils.RAMDatasetWrapper(dataset=train_dataset_, transform=transforms) # Shall be: torch.Size([1, 64, 64])
val_dataset = utils.RAMDatasetWrapper(dataset=val_dataset_, transform=transforms)

# Display some images after transformation to tensor
utils.plot_CT_scans(dataset=train_dataset, n_images=6, label=0, gray=True)
utils.plot_CT_scans(dataset=train_dataset, n_images=6, label=1, gray=True)


# Define train parameters
batch_size = 32
num_workers = 0 # Set to 2 or 4 when usign GPU
nEpochs = 2

# Setup Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# User Input: Model type and task
model_typ = input("Enter model type (mlp/cnn): ").lower()
if model_typ not in ['mlp', 'cnn']:
    print("Invalid model type. Please enter one of the given models.")
    exit()

task = input("Enter task (train/regularization): ").lower()
if task not in ['train', 'regularization']:
    print("Invalid task. Please enter one of the given tasks.")
    exit()

# Initialize model based on the user input
if model_typ == 'mlp':
    model = utils.MLPModel(target_size * target_size * nChannel, 64)
elif model_typ == 'cnn':
    model = utils.CNNModel(nChannel=nChannel, nClasses=2)

# Setting model to the specified device above (GPU/CPU)
model = model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# If Task = train, train the model
if task == 'train':
    train_loss, val_loss, train_acc, val_acc = utils.train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        nEpochs,
        loss_fn,
        device,
        weight=True if model_typ == 'mlp' else None
    )

    # Training plot result
    utils.plot(train_loss, val_loss, train_acc, val_acc, title=f'standard {model_typ.upper()}')

    # Validation metrics
    val_accuracy = val_acc[-1]
    val_loss_final = val_loss[-1]
    print(f'No regularization. Validation loss: {val_loss_final:.4f}, validation accuracy: {val_accuracy:.4f}')

# If task = regularization, apply regularization
elif task == "regularization":
    regularization_typ = input("Enter the regularization method (early_stopping/l2/data_augmentation): ").lower()
    if regularization_typ not in ['early_stopping', 'l2', 'data_augmentation']:
        print("Invalid regularization. Please enter one of the mentioned methods.")
        exit()

    # Early stopping regularization
    if regularization_typ == 'early_stopping':
        train_loss, val_loss, train_acc, val_acc = utils.train_early_stopping(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            nEpochs,
            loss_fn,
            device
        )

        # Early stopping plotting results
        utils.plot(train_loss, val_loss, train_acc, val_acc, title=f'{model_typ.upper()} - Early Stopping')
        print(f'CNN - Early stopping. Validation loss: {val_loss[-1]:.4f}, validation accuracy: {val_acc[-1]:.4f}')

    # L2 regularization
    elif regularization_typ == 'l2':
        alpha = 0.00005
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
        train_loss, val_loss, train_acc, val_acc = utils.train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            nEpochs,
            loss_fn,
            device,
            weight=None
        )

        # L2 Plotting results
        utils.plot(train_loss, val_loss, train_acc, val_acc, title=f'{model_typ.upper()} - L2 regularization')
        print(f'L2 regularization. Validation loss: {val_loss[-1]:.4f}, validation accuracy: {val_acc[-1]:.4f}')

    # Data augmentation
    elif regularization_typ == 'data_augmentation':
        from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomAffine, RandomPerspective
        transforms_da = Compose([
            RandomHorizontalFlip(0.4),
            RandomRotation(degrees=7),
            RandomAffine(degrees=7),
            RandomPerspective(0.4)
        ])
        augmented_train_dataset = utils.DatasetFromSubset(train_dataset, transforms_da)
        train_dataloader_augmented = DataLoader(augmented_train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        train_loss, val_loss, train_acc, val_acc = utils.train(
            model,
            train_dataloader=train_dataloader_augmented,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            n_epochs=nEpochs,
            loss_fn=loss_fn,
            device=device,
            weight=None
        )

        # Data augmentation plotting results
        utils.plot(train_loss, val_loss, train_acc, val_acc, title=f'{model_typ.upper()} - Data augmentation')
        print(f'Data augmentation. Validation loss: {val_loss[-1]:.4f}, validation accuracy: {val_acc[-1]:.4f}')

# We compare results and summarise
acc_curves = [train_acc, val_acc]
reg_typ = ["train", "validation"]

# Plotting accuracy curves for different tasks
for acc_curve in acc_curves:
    plt.plot(np.arange(len(acc_curves)), acc_curves)
plt.title('Accuracy curves')
plt.legend(reg_typ)

# Display end validation accuracies for the chosen task
table_data = [
    ['Model', 'Validation Loss', 'Validation accuracy'],
    [f'{model_typ.upper()} - {task}', val_loss[-1], val_acc[-1]]
]

print("{: >30}| {: >30}| {: >30}".format(*table_data[0]))
print('-'.join(['' for i in range(95)]))
for row in table_data[1:]:
    print("{: >30}| {:30.4f}| {: >30}".format(*row))