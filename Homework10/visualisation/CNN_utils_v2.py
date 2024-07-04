import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
from random import sample
#from tqdm.notebook import tqdm # When running in browser
from tqdm.autonotebook import tqdm



# Datawrapper
class RAMDatasetWrapper(torch.utils.data.Dataset):
    """
    Initialization by taking dataset as input and allowing to transform it for preprocessing.
    """
    def __init__(self, dataset, transform=None):
        data = []
        for sample in tqdm(dataset): # Displays process bar while filling in data list
            data.append(sample)
        self.n = len(data)
        self.data = data
        self.transform = transform 
        
    def __getitem__(self, ind):
        if self.transform is not None: # Optional transformation 
            return self.transform(self.data[ind][0]), self.data[ind][1]
        else:
            return self.data[ind] # otherwise returns original data
    
    def set_tranform(self, transform): # Allows to set or change transformation after obj init
        self.transform = transform
    
    def __len__(self):
        return self.n
    

# Plot function to plot images as torch.Tensor or PIL.Image.Image
def plot_CT_scans(dataset, n_images=1, label=1, gray=False):
    """
****************Function plot_CT_scans****************
Input:
           dataset    - object which is accessible like ImageFolder (https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder)
                        either Object[torch.Tensor][label] or Object[PIL.Image.Image][label]
           n_images   - number of random images which will be displayed from the dataset
           label      - Selecton whether samples of healthy patients (label=0) or samples of patients with Penumonia shall be displayed
           gray       - Bool value whether images shall be displayed in grayscale or not (alternative RGB)  

Return: none       - This functions allows to plot the CT scans in this specific setting
    """

# ***Ensure correctness of dataset type
# Case 1 - torch.Tensor
    if isinstance(dataset[0][0], torch.Tensor):
        print("\nInput format: Tensor --> convert to PIL image before plotting\n")
        to_PIL_transformer = ToPILImage()
    
    # ***Generate label
        if label == 0:
            print("Show {} random images of healthy patients from dataset".format(n_images))
        elif label == 1:
            print("Show {} random images of patients with Pneumonia from dataset".format(n_images))
        else:
            raise ValueError("Something went wrong with the label - must be\n0 - healthy\n1 - pneumonia")

        # Filter dataset based on the label
        # Enumerate along tuples (_, lbl) only if lbl matches with label
        filtered_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
        
        # Randomly select num_images indices from the filtered subset
        selected_indices = sample(filtered_indices, 
                                min(n_images, 
                                    len(filtered_indices)))

    # ***Plot
        if n_images >= 1 or n_images <= 6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,4))
            for idx, ax in zip(selected_indices, axes.flat):
                if gray:
                    ax.imshow(to_PIL_transformer(dataset[idx][0]), cmap="gray")
                    ax.axis("off")
                else:
                    ax.imshow(to_PIL_transformer(dataset[idx][0]))
                    ax.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Number of images to plot out of range - must be between 1 - 5")

# ***Ensure correctness of dataset type
# Case 2 PIL.Image.Image
    elif isinstance(dataset[0][0], PIL.Image.Image):
        print("\nAlready PIL image --> Plot\n") 
            # ***Generate label
        if label == 0:
            print("Show {} random images of healthy patients from dataset".format(n_images))
        elif label == 1:
            print("Show {} random images of patients with Pneumonia from dataset".format(n_images))
        else:
            raise ValueError("Something went wrong with the label - must be\n0 - healthy\n1 - pneumonia")

        # Filter dataset based on the label
        # Enumerate along tuples (_, lbl) only if lbl matches with label
        filtered_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
        
        # Randomly select num_images indices from the filtered subset
        selected_indices = sample(filtered_indices, 
                                min(n_images, 
                                    len(filtered_indices)))
    # ***Plot
        if n_images >= 1 or n_images <= 6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,4))
            for idx, ax in zip(selected_indices, axes.flat):
                if gray:
                    ax.imshow(dataset[idx][0], cmap="gray")
                    ax.axis("off")
                else:
                    ax.imshow(dataset[idx][0])
                    ax.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Number of images to plot out of range - must be between 1 - 5") 
    else:
        raise TypeError("Unsupported dataset format. Supported are torch.Tensor or PIL.Image.Image\n ensure dataset can be accessed with format dataset[x][y].")
    
          
# Plot model results
# Losses
def plot(train_losses, val_losses, train_accuracies, val_accuracies, title):
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('{}: Train/val loss'.format(title))
    plt.show()

# Accuracies
    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies)
    plt.plot(np.arange(len(val_accuracies)), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('{}: Train/val accuracy'.format(title))
    plt.show()


# Multi Layer Preceptron Model inheriting from nn.Module
# Input: input_dim, hidden_dim
# Output: Binary classification; torch.tensor([input.size(0), -1])    
class MLPModel(nn.Module):
    # nn.Linear() - linear trnsformation  to incoming data y = xA.T + b (uses TensorFloat32)
    # Input dim defines dimensionality of the input
    # input_dim: size of each input sample
    # hidden_dim: size of each output sample
    # Output dim defines dimensionality of the output
    # when calling function

    # Define constructor
    def __init__(self, input_dim, hidden_dim):
        # Super constructor call to initialize MLPModel object as nn.Module instance
        # Fully connected network with defined input/output dimensions - 3 hidden layers
        # nn.ReLU() - Rectified Linear Unit activation function (introduces non-linearity to model)
        # bias: If set to ``False``, the layer will not learn an additive bias.
        # Default: ``True``
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
           
    # Forward method
    def forward(self, input):
        # Reshape input (view similar as reshape)
        # Applies layers which were previously defined in the constructor
        return self.layers(input.view(input.size(0), -1))


# Convolutional Neural Network inheriting from nn.Module
# Input: None
# Output: Binary classification
class CNNModel(nn.Module):

    # Input
    # nChannels: number of channels of input image (1 if grayscale, 3 for RGB)
    # nClasses: number of classes for classification task    
    def __init__(self, nChannel, nClasses):

            # Stride defines the pixel stepsize of the filter movement - increase for reducing computational costs
            # MaxPooling: divides input tensor into non-overlapping regions and outputs value with highest activation (max value)
            #             It retains the most salient features which most affect the conv filters and inherit features or patterns.
            #             The max value is not same as the most important feature (depends on application / task). 

        super(CNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=nChannel, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # uses ceil instead of floor to cimpute output shape
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),

            # Output channel size depending on number of classes 
            nn.Conv2d(in_channels=256, out_channels=nClasses, kernel_size=1, stride=1)
        )        

    def forward(self, input):
        return self.layers(input).view(input.size(0), -1)


# Training per model
def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_fn, device=None, weight=None):

    # Monitor loss functions as the training progresses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    model = model.to(device) # Check if this fixes the two different devices

    for epoch in range(n_epochs):
        
        model.train() # Set to train mode - allowing to change parameters etc
        train_loss, train_accuracy = train_epoch(model, 
                                                 train_dataloader, 
                                                 optimizer, 
                                                 loss_fn, 
                                                 device, 
                                                 weight=weight)
        
        model.eval() # Set to evaluation mode - no updates on parameters and reclaculations of gradients
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device, weight=weight)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                      train_losses[-1],
                                                                                                      train_accuracies[-1],
                                                                                                      val_losses[-1],
                                                                                                      val_accuracies[-1]))
    return train_losses, val_losses, train_accuracies, val_accuracies            

'''
# Model training per epoch
def train_epoch(model, train_dataloader, optimizer, loss_fn, device, weight=None):
    losses = []
    correct_predictions = 0
    model = model.to(device) # Check if this fixes the issue with the two devices    


    # Iterate mini batches over training dataset
    for images, labels in tqdm(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Run predictions
        output = model(images)
        
        # Set gradients to zero
        optimizer.zero_grad()
        
        # Compute loss
        if weight is not None:
            curr_weight = weight_calculation(labels).to(device) # Weight can be added for weighting betw. the classes
            loss_fn = nn.CrossEntropyLoss(weight=curr_weight)
        loss = loss_fn(output, labels)
        
        # Backpropagate (compute gradients)
        loss.backward()
        
        # Make an optimization step (update parameters)
        optimizer.step()
        
        # Log metrics
        losses.append(loss.item())
        predicted_labels = output.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
    accuracy = 100.0 * correct_predictions / len(train_dataloader.dataset)
    # Return loss values for each iteration and accuracy
    mean_loss = np.array(losses).mean()
    return mean_loss, accuracy



# Model evaluation
def evaluate(model, dataloader, loss_fn, device, weight=None):
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Run predictions
            output = model(images).to(device)

            # Compute loss
            if weight is not None:
                curr_weight = weight_calculation(labels).to(device) # Weight can be added for weighting betw. the classes
                loss_fn = nn.CrossEntropyLoss(weight=curr_weight)
            loss = loss_fn(output, labels)

            # Save metrics
            predicted_labels = output.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            losses.append(loss.item())
   
    mean_loss = np.array(losses).mean()
    accuracy = 100.0 * correct_predictions / len(dataloader.dataset)

    # Return mean loss and accuracy
    return mean_loss, accuracy


'''
#------------------- train_epoch new, to include class distribution 

def train_epoch(model, dataloader, optimizer, loss_fn, device=None, weight=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_distribution = {'healthy': 0, 'pneumonia': 0}
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Count class distribution
        class_distribution['healthy'] += (targets == 0).sum().item()
        class_distribution['pneumonia'] += (targets == 1).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, class_distribution

def evaluate(model, dataloader, loss_fn, device=None, weight=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_distribution = {'healthy': 0, 'pneumonia': 0}
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Count class distribution
            class_distribution['healthy'] += (targets == 0).sum().item()
            class_distribution['pneumonia'] += (targets == 1).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, class_distribution

#-------------------

def weight_calculation(labels):
    n_class0 = torch.sum(labels == 0).item()
    n_class1 = torch.sum(labels == 1).item()
    
    # Calculate class weights
    weight_class0 = (n_class0 + n_class1) / (2.0 * n_class0)  if n_class0 != 0 else 1
    weight_class1 = (n_class0 + n_class1) / (2.0 * n_class1)  if n_class1 != 0 else 1

    # Normalize weights
    total_classes = 2  # Assuming binary classification in this case
    normalized_weight_class0 = 1 / (total_classes * weight_class0)
    normalized_weight_class1 = 1 / (total_classes * weight_class1)

    # Define the class weights
    class_weights = torch.tensor([normalized_weight_class0, normalized_weight_class1])
    return class_weights



#**********************************************************************************
#********************************* Regularization *********************************
#**********************************************************************************

# Early stopping
from copy import deepcopy

def train_early_stopping(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_fn, device):
    # We will monitor loss functions as the training progresses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Inizializaiton of early stopping
    best_val_loss = np.inf
    best_model = None
    patience = 5 # if no improvement after 5 epochs, stop training
    counter = 0
    
    model = model.to(device) # Check if this fixes the issue with the two devices

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_accuracy = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            loss_fn, 
            device)
        
        model.eval()
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                      train_losses[-1],
                                                                                                      train_accuracies[-1],
                                                                                                      val_losses[-1],
                                                                                                      val_accuracies[-1]))
        ### Early stopping conditions 
        if val_loss < best_val_loss: # when validation loss was improved, save model and reset counter
            best_val_loss = val_loss
            best_model = deepcopy(model) # Save the best model
            counter = 0
        else: # If validation loss not improved, increase the counter
            counter += 1
        if counter == patience: # If no improvement for some number of epochs, stop training
            print('No improvement for {} epochs; training stopped.'.format(patience))
            break
    
    # Copy best model parameters to model
    for param1, param2 in zip(model.parameters(), best_model.parameters()):
        param1.data = param2.data

    return train_losses, val_losses, train_accuracies, val_accuracies


# Data augmentation
class DatasetFromSubset(Dataset):

    # Define constructor with subset 
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)












