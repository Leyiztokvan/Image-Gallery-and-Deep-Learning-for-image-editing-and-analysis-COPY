import matplotlib.pyplot as plt
import numpy as np
import os

# Save the plot to a given folder with a given name
def save_plot(fig: plt.Figure, save_dir: str, img_name: str = 'image'):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, img_name))
    plt.close(fig)  # Close the figure after saving to avoid overlap


# Function to save training results
def save_results(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'val_losses.npy'), val_losses)
    np.save(os.path.join(save_dir, 'train_accuracies.npy'), train_accuracies)
    np.save(os.path.join(save_dir, 'val_accuracies.npy'), val_accuracies)

# Function to load training results
def load_results(save_dir):
    train_losses = np.load(os.path.join(save_dir, 'train_losses.npy'))
    val_losses = np.load(os.path.join(save_dir, 'val_losses.npy'))
    train_accuracies = np.load(os.path.join(save_dir, 'train_accuracies.npy'))
    val_accuracies = np.load(os.path.join(save_dir, 'val_accuracies.npy'))
    return train_losses, val_losses, train_accuracies, val_accuracies


# Function to save class distribution data
def save_class_distribution(class_distribution, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, file_name), class_distribution)

# Function to load class distribution data
def load_class_distribution(save_dir, file_name):
    file_path = os.path.join(save_dir, file_name + '.npy')
    return np.load(file_path, allow_pickle=True).item()