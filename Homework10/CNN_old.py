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
utils.plot_CT_scans(dataset=train_dataset_, n_images=1, label=1, gray=True)
#utils.plot_CT_scans(dataset=train_dataset_, n_images=6, label=1, gray=True)


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


#### Implement Multi-Layer-Preceptron (MLP) model
# Model defined and documented in the utils file
mlp_model = utils.MLPModel(target_size * target_size * nChannel, 64)

#### Implement the CNN model
# Model defined and documented in the utils file
cnn_model = utils.CNNModel(nChannel=nChannel, nClasses=2)

#### Implement the training code
# Train the MLP model

# Define parameters, optimizer, and loss function for MLP model and set the model to device (GPU or CPU)
mlp_model = mlp_model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(mlp_model.parameters(), 
                             lr=learning_rate,
                             betas=[0.9, 0.999],
                             eps=1e-8,
                             weight_decay=0.,
                             amsgrad=False, 
                             foreach=None,
                             maximize=False) 
n_epochs = nEpochs
loss_fn = nn.CrossEntropyLoss()

# Trainin MLP model and access for training and validation the loss and accuracy
mlp_train_loss, mlp_val_loss, mlp_train_acc, mlp_val_acc = utils.train(
    mlp_model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    n_epochs, 
    loss_fn, 
    device,
    weight=True)


#### Train the CNN model
# Define parameters, optimizer, and loss function for CNN model and set the model to device (GPU or CPU)
cnn_model = cnn_model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
n_epochs = nEpochs
loss_fn = nn.CrossEntropyLoss()

#### Trainin CNN model and access for training and validation the loss and accuracy
cnn_train_loss, cnn_val_loss, cnn_train_acc, cnn_val_acc = utils.train(
    cnn_model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    n_epochs, 
    loss_fn, 
    device,
    weight=None)



#### MLP model - plot and show results
utils.plot(mlp_train_loss, 
           mlp_val_loss, 
           mlp_train_acc, 
           mlp_val_acc, 
           title='standard MLP')

mlp_val_accuracy_no_regularization = mlp_val_acc[-1]
mlp_val_loss_no_regularization = mlp_val_loss[-1]
print('No regularization. Validation loss: {:.4f}, validation accuracy: {:.4f}'.format(mlp_val_loss_no_regularization, mlp_val_accuracy_no_regularization))


#### CNN Model - plot and show results
utils.plot(
    cnn_train_loss, 
       cnn_val_loss, 
       cnn_train_acc, 
       cnn_val_acc, 
       title='standard CNN')

cnn_val_accuracy_no_regularization = cnn_val_acc[-1]
cnn_val_loss_no_regularization = cnn_val_loss[-1]
print('No regularization. Validation loss: {:.4f}, validation accuracy: {:.4f}'.format(cnn_val_loss_no_regularization, cnn_val_accuracy_no_regularization))


### Implement  regularization techniques
# First technique - Early stopping
# Set up CNN model with early stopping regularization and train it
cnn_model_early_stop = utils.CNNModel(nChannel=nChannel, nClasses=2)
cnn_model_early_stop = cnn_model_early_stop.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(cnn_model_early_stop.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

train_losses_es, val_losses_es, train_acc_es, val_acc_es = utils.train_early_stopping(
    cnn_model_early_stop, 
    train_dataloader,
    val_dataloader, 
    optimizer, 
    nEpochs, 
    loss_fn,
    device)


#### CNN Model with early stopping regularization - plot and show results
utils.plot(train_losses_es, 
           val_losses_es, 
           train_acc_es, 
           val_acc_es, 
           title='CNN - Early Stopping')

val_loss_es = val_losses_es[-1]
val_accuracy_es = val_acc_es[-1]
print('CNN - Early stopping. Validation loss: {:.4f}, validation accuracy: {:.4f}'.format(val_loss_es, val_accuracy_es))


### Second technique - L2 regularization aka weight decay
# We can specify strength of weight decay in optimizer (parameter of optimizer's constructor; https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)
cnn_model_L2 = utils.CNNModel(nChannel, 2)
cnn_model_L2 = cnn_model_L2.to(device)
learning_rate = 0.001

# L2 regularization code
alpha = 0.00005 # If too big the network won't train at all (weights will be zero)
optimizer = torch.optim.Adam(cnn_model_L2.parameters(), lr=learning_rate, weight_decay=alpha)
n_epochs = nEpochs
loss_fn = nn.CrossEntropyLoss()

train_losses_wd, val_losses_wd, train_acc_wd, val_acc_wd = utils.train(
    cnn_model_L2, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    n_epochs, 
    loss_fn,
    device,
    weight=None)

utils.plot(train_losses_wd, val_losses_wd, train_acc_wd, val_acc_wd, title=' CNN - L2 regularization')

val_loss_weightdecay  = val_losses_wd[-1] 
val_accuracy_weightdecay = val_acc_wd[-1]
print('L2 regularization. Validation loss: {:.4f}, validation accuracy: {:.4f}'.format(val_loss_weightdecay, val_accuracy_weightdecay))

#### Third technique - Data augmentation (additional) 
# We will add several transformations to increase variability in the training data while preserving the original content and label (https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image):
# - Random horizontal flip (Horizontally flip the input with a given probability)
# - Random rotation (Rotate the input by angle)
# - Random affine (Random affine transformation the input keeping center invariant)
# *Previous transformations during wraping remain preserved in the datasets and are therefore not needed again

from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomAffine, RandomPerspective
transforms_da = Compose([
                        RandomHorizontalFlip(0.4),
                        RandomRotation(degrees=7),
                        RandomAffine(degrees=7),
                        RandomPerspective(0.4)
                    ])

# Apply to train dataset data augmentation
augmented_train_dataset = utils.DatasetFromSubset(train_dataset, transforms_da)
train_dataloader_augmented = DataLoader(augmented_train_dataset, batch_size, shuffle=True, num_workers=num_workers)


# Train with augmented images
augmentation_cnn_model = utils.CNNModel(nChannel, 2)
augmentation_cnn_model = augmentation_cnn_model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(augmentation_cnn_model.parameters(), lr=learning_rate)
n_epochs = nEpochs
loss_fn = nn.CrossEntropyLoss()

train_losses_da, val_losses_da, train_acc_da, val_acc_da = utils.train(
    model=augmentation_cnn_model,
    train_dataloader=train_dataloader_augmented,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    n_epochs=nEpochs,
    loss_fn = loss_fn,
    device=device,
    weight=None)


# Plot results
utils.plot(train_losses_da, val_losses_da, train_acc_da, val_acc_da, title='Data augmentation')

val_loss_dataaugmentation = val_losses_da[-1]
val_accuracy_dataaugmentation = val_acc_da[-1]
print('Data augmentation. Validation loss: {:.4f}, validation accuracy: {:.4f}'.format(val_loss_dataaugmentation, val_accuracy_dataaugmentation))


# # Results comparison - Summary
accuracy_curves = [mlp_val_acc, cnn_val_acc, val_acc_es, val_acc_da, val_acc_wd]
reg_types = ["mlp - no regularization", "CNN - no regularization", "early stopping", "data augmentation", "L2 regularization"]

for accuracy_curve in accuracy_curves:
    plt.plot(np.arange(len(accuracy_curve)), accuracy_curve)
plt.title('Validation accuracy curves for different regularization')
plt.legend(reg_types)


# ### Final validation accuracies for the following scenarios:
# - Fully connected neural network (MPL)
# - Convolutional neural network (CNN)
# - CNN with early stopping regularization
# - CNN with L2 regularization
# - CNN with data augmentation 
table_data = [
    ['Model', 'Validation Loss', 'Validation accuracy'],
    ["MLP - No regularization", mlp_val_loss_no_regularization, mlp_val_accuracy_no_regularization],
    ['CNN - No regularization', cnn_val_loss_no_regularization, cnn_val_accuracy_no_regularization], 
    ['CNN - Early stopping', val_loss_es, val_accuracy_es],
    ['CNN - L2 regularization', val_loss_weightdecay, val_accuracy_weightdecay],
    ['CNN - Data augmentation', val_loss_dataaugmentation, val_accuracy_dataaugmentation]
]
print("{: >30}| {: >30}| {: >30}".format(*table_data[0]))
print('-'.join(['' for i in range(95)]))
for row in table_data[1:]:
    print("{: >30}| {:30.4f}| {: >30}".format(*row))
