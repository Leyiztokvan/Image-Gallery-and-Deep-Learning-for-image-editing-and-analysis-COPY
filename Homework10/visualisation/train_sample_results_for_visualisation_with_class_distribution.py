# Imports
import torch 
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder 
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale
import torch.nn as nn

import CNN_utils_v2 as utils 
import plot_utils_v2 as plot_utils

def train_with_class_distribution(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_fn, device=None, weight=None):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_class_distribution = {'healthy': [], 'pneumonia': []}
        val_class_distribution = {'healthy': [], 'pneumonia': []}
        
        for epoch in range(n_epochs):
            model.train()
            train_loss, train_accuracy, train_distribution = utils.train_epoch(model, train_dataloader, optimizer, loss_fn, device, weight=weight)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_class_distribution['healthy'].append(train_distribution['healthy'])
            train_class_distribution['pneumonia'].append(train_distribution['pneumonia'])
            
            model.eval()
            val_loss, val_accuracy, val_distribution = utils.evaluate(model, val_dataloader, loss_fn, device, weight=weight)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_class_distribution['healthy'].append(val_distribution['healthy'])
            val_class_distribution['pneumonia'].append(val_distribution['pneumonia'])
            
            print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                          train_losses[-1],
                                                                                                          train_accuracies[-1],
                                                                                                          val_losses[-1],
                                                                                                          val_accuracies[-1]))
        
        # Save class distribution data
        save_dir = "Homework10/visualisation/training_results_sample_with_class_distribution/class_distribution"
        plot_utils.save_class_distribution(train_class_distribution, save_dir, file_name="class_distribution_train.npy")
        plot_utils.save_class_distribution(val_class_distribution, save_dir, file_name="class_distribution_val.npy")

        return train_losses, val_losses, train_accuracies, val_accuracies


# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("We are using {}".format(device))

# Read data

train_dir = "xray_dataset/chest_xray_64/train"
val_dir = "xray_dataset/chest_xray_64/val"
train_dataset_ = ImageFolder(root=train_dir, is_valid_file=lambda x: x.endswith("jpeg"))
val_dataset_ = ImageFolder(root=val_dir, is_valid_file=lambda x: x.endswith("jpeg"))

# Setup composition of transforms
target_size = 64
nChannel = 1
transforms = Compose([
    Grayscale(1),
    ToTensor(),
    Normalize(mean=(0.5), std=(0.5)),
])

# Wrap datasets with transformations
train_dataset = utils.RAMDatasetWrapper(dataset=train_dataset_, transform=transforms)
val_dataset = utils.RAMDatasetWrapper(dataset=val_dataset_, transform=transforms)

# Define train parameters
batch_size = 32
num_workers = 0
nEpochs = 10

# Setup Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Implement Multi-Layer-Preceptron (MLP) model
mlp_model = utils.MLPModel(target_size * target_size * nChannel, 64)

# Train the MLP model with class distribution collection
mlp_model = mlp_model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
mlp_train_loss, mlp_val_loss, mlp_train_acc, mlp_val_acc = train_with_class_distribution(
    mlp_model, train_dataloader, val_dataloader, optimizer, nEpochs, loss_fn, device)

# Implement the CNN model
cnn_model = utils.CNNModel(nChannel=nChannel, nClasses=2)

# Train the CNN model with class distribution collection
cnn_model = cnn_model.to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
cnn_train_loss, cnn_val_loss, cnn_train_acc, cnn_val_acc = train_with_class_distribution(
    cnn_model, train_dataloader, val_dataloader, optimizer, nEpochs, loss_fn, device)


# Save the results
# Set the directories where the results are saved
save_dir_mlp = "Homework10/visualisation/training_results_sample_with_class_distribution/mlp"
save_dir_cnn = "Homework10/visualisation/training_results_sample_with_class_distribution/cnn"

plot_utils.save_results(mlp_train_loss, mlp_val_loss, mlp_train_acc, mlp_val_acc, save_dir_mlp)
plot_utils.save_results(cnn_train_loss, cnn_val_loss, cnn_train_acc, cnn_val_acc, save_dir_cnn)