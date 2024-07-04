import numpy as np
import matplotlib.pyplot as plt
import os

import plot_utils_v2 as plot_utils

def assess_model(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Assess model performance based on loss and accuracy.

    Parameters:
    - train_losses (list): List of training losses.
    - val_losses (list): List of validation losses.
    - train_accuracies (list): List of training accuracies.
    - val_accuracies (list): List of validation accuracies.
    """
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_train_accuracy = train_accuracies[-1]
    final_val_accuracy = val_accuracies[-1]
    
    if final_train_loss < final_val_loss and final_train_accuracy > final_val_accuracy:
        print("The model is likely overfitting.")
    elif final_train_loss > final_val_loss and final_train_accuracy < final_val_accuracy:
        print("The model is likely underfitting.")
    else:
        print("The model is performing well.")

    bias = 1 - final_train_accuracy
    variance = final_val_accuracy - final_train_accuracy

    print(f"Bias: {bias:.4f}")
    print(f"Variance: {variance:.4f}")



# Set the directories
load_dir_cnn = "Homework10/visualisation/training_results_sample_with_class_distribution/cnn"
load_dir_mlp = "Homework10/visualisation/training_results_sample_with_class_distribution/mlp"
save_dir = "Homework10/visualisation/plots/analysis"

# Load the saved results
train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn = plot_utils.load_results(load_dir_cnn)
train_losses_mlp, val_losses_mlp, train_accuracies_mlp, val_accuracies_mlp = plot_utils.load_results(load_dir_mlp)

# Assess CNN model performance
assess_model(train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn)
# Assess MLP model performance
assess_model(train_losses_mlp, val_losses_mlp, train_accuracies_mlp, val_accuracies_mlp)



# creates a comprehensive plot that includes learning curves along with bias and variance information, 
# by combining several elements into one figure. A light gray reference curve representing a well-fitting model
def plot_learning_curves_with_analysis(subplots_arrays: list, reference_curve: dict, show_plot: bool = True, plot_title: str = "Plot Title", subplots_titles: list = ['Subplot 1', 'Subplot 2'], subplots_labels: list =[['Subplot 1 Label X ax', 'Subplot 1 Label Y ax'], ['Subplot 2 Label X ax', 'Subplot 2 Label Y ax']], subplots_array_labels: list =[['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']], fig_size: list = [14, 10], axis: str = 'on') -> plt.Figure:
    """
    Plot learning curves with analysis.

    Parameters:
    - subplots_arrays (list): List of lists containing data for subplots. Each sublist should contain two lists: [data1, data2].
    - reference_curve (dict): Dictionary containing reference loss and accuracy curves.
    - show_plot (bool): Boolean to show the plot. Default is True.
    - plot_title (str): Title for the entire plot. Default is 'Plot Title'.
    - subplots_titles (list): Titles for each subplot. Default is ['Subplot 1', 'Subplot 2'].
    - subplots_labels (list): Labels for x and y axes for each subplot. Default is [['Subplot 1 Label X ax', 'Subplot 1 Label Y ax'], ['Subplot 2 Label X ax', 'Subplot 2 Label Y ax']].
    - subplots_array_labels (list): Labels for each array in the subplots. Default is [['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']].
    - fig_size (list): Dimension of the figure. Default is [14, 10].
    - axis (str): Axis of the subplots. Default is "on".

    Returns:
    - matplotlib.pyplot.Figure object
    """

    epochs = range(1, len(subplots_arrays[0][0]) + 1)

    # Calculate bias and variance
    final_train_accuracy = subplots_arrays[1][0][-1]
    final_val_accuracy = subplots_arrays[1][1][-1]

    bias = 1 - final_train_accuracy
    variance = final_val_accuracy - final_train_accuracy

    # Determine the relative value to display (variance/bias)
    relative_value = variance / bias if bias != 0 else float('inf')

    # Create a new figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(fig_size[0], fig_size[1]))

    # Plot data on the first subplot
    ax[0].plot(epochs, subplots_arrays[0][0], label=subplots_array_labels[0][0], color='blue')
    ax[0].plot(epochs, subplots_arrays[0][1], label=subplots_array_labels[0][1], color='orange')
    ax[0].plot(epochs, reference_curve['loss'], label='Reference Model Loss', color='lightgray', linestyle='--')
    ax[0].set_title(subplots_titles[0])
    ax[0].set_xlabel(subplots_labels[0][0])
    ax[0].set_ylabel(subplots_labels[0][1])
    ax[0].legend()
    ax[0].axis(axis)

    # Plot data on the second subplot
    ax[1].plot(epochs, subplots_arrays[1][0], label=subplots_array_labels[1][0], color='blue')
    ax[1].plot(epochs, subplots_arrays[1][1], label=subplots_array_labels[1][1], color='orange')
    ax[1].plot(epochs, reference_curve['accuracy'], label='Reference Model Accuracy', color='lightgray', linestyle='--')
    ax[1].set_title(subplots_titles[1])
    ax[1].set_xlabel(subplots_labels[1][0])
    ax[1].set_ylabel(subplots_labels[1][1])
    ax[1].legend()
    ax[1].axis(axis)

    # Annotate bias and variance
    plt.figtext(0.82, 0.92, f"Bias: {bias:.4f}\nVariance: {variance:.4f}\nRelative Value (Variance/Bias): {relative_value:.4f}",
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    # Add a label to the entire plot
    fig.suptitle(plot_title, fontsize=16)

    # Adjust the layout to make more space for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot 
    if show_plot:
        plt.show()

    return fig



# Define CNN subplot data
cnn_arrays = [
    [train_losses_cnn, val_losses_cnn],
    [train_accuracies_cnn, val_accuracies_cnn]
]

# Generate a reference curve (this is just a placeholder, replace with actual reference values if available)
reference_curve = {
    'loss': np.linspace(0.1, 0.5, len(train_losses_cnn)),  # Example reference loss curve
    'accuracy': np.linspace(0.95, 0.8, len(train_accuracies_cnn))  # Example reference accuracy curve
}


# Plot the learning curves of the CNN model with analysis
cnn_analysis_fig = plot_learning_curves_with_analysis(subplots_arrays=cnn_arrays, reference_curve=reference_curve, plot_title='CNN Model Learning Curves with Analysis', subplots_titles=['Training and Validation Losses', 'Training and Validation Accuracies'], subplots_labels=[['Epochs', 'Loss'], ['Epochs', 'Accuracy']], subplots_array_labels=[['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']], fig_size=[14, 10], axis='on')

plot_utils.save_plot(cnn_analysis_fig, img_name='CNN Model Learning Curves with Analysis', save_dir=save_dir)

# Define MLP subplot data
mlp_arrays = [
    [train_losses_mlp, val_losses_mlp],
    [train_accuracies_mlp, val_accuracies_mlp]
]

# Generate a reference curve (this is just a placeholder, replace with actual reference values if available)
reference_curve = {
    'loss': np.linspace(0.1, 0.5, len(train_losses_mlp)),  # Example reference loss curve
    'accuracy': np.linspace(0.95, 0.8, len(train_accuracies_mlp))  # Example reference accuracy curve
}


# Plot the learning curves of the MLP model with analysis
mlp_analysis_fig = plot_learning_curves_with_analysis(subplots_arrays=mlp_arrays, reference_curve=reference_curve, plot_title='MLP Model Learning Curves with Analysis', subplots_titles=['Training and Validation Losses', 'Training and Validation Accuracies'], subplots_labels=[['Epochs', 'Loss'], ['Epochs', 'Accuracy']], subplots_array_labels=[['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']], fig_size=[14, 10], axis='on')

plot_utils.save_plot(mlp_analysis_fig, img_name='MLP Model Learning Curves with Analysis', save_dir=save_dir)




def plot_class_distribution(train_distribution, val_distribution, save_dir: str, image_name: str):
    """
    Function to track sample distribution e.g. for batch variation, cross-validation, leave-one-out-cross-validation"

    Parameters:
    - train_distribution (dict): Dictionary containing training class distribution data.
    - val_distribution (dict): Dictionary containing validation class distribution data.
    - save_dir (str): Directory to save the plot image.
    - image_name (str): Name of the image file.

    Returns:
    None
    """
    epochs = np.arange(len(train_distribution['healthy']))
    bar_width = 0.35
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot training data
    axs[0].bar(epochs - bar_width / 2, train_distribution['healthy'], bar_width, label='Healthy', alpha=0.7, color='blue')
    axs[0].bar(epochs + bar_width / 2, train_distribution['pneumonia'], bar_width, label='Pneumonia', alpha=0.7, color='orange')
    axs[0].set_title('Training Data')
    axs[0].legend()
    
    # Add annotations for training data
    for i, count in enumerate(train_distribution['healthy']):
        axs[0].annotate(f'{count}', xy=(epochs[i] - bar_width / 2, count), xytext=(-10, 3), textcoords='offset points', ha='center', fontsize=8)
    for i, count in enumerate(train_distribution['pneumonia']):
        axs[0].annotate(f'{count}', xy=(epochs[i] + bar_width / 2, count), xytext=(10, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Plot validation data
    axs[1].bar(epochs - bar_width / 2, val_distribution['healthy'], bar_width, label='Healthy')
    axs[1].bar(epochs + bar_width / 2, val_distribution['pneumonia'], bar_width, label='Pneumonia')
    axs[1].set_title('Validation Data')
    axs[1].legend()
    
    # Add annotations
    for i, count in enumerate(val_distribution['healthy']):
        axs[1].annotate(f'{count}', xy=(epochs[i] - bar_width / 2, count), xytext=(-10, 3), textcoords='offset points', ha='center', fontsize=8)
    for i, count in enumerate(val_distribution['pneumonia']):
        axs[1].annotate(f'{count}', xy=(epochs[i] + bar_width / 2, count), xytext=(10, 3), textcoords='offset points', ha='center', fontsize=8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.suptitle('Class Distribution Over Epochs')
    
    plt_path = os.path.join(save_dir, image_name)
    plt.savefig(plt_path)
    plt.show()
    plt.close()

# Usage example
load_dir = "Homework10/visualisation/training_results_sample_with_class_distribution/class_distribution"

# Load the class distribution data for training and validation
train_class_distribution = plot_utils.load_class_distribution(load_dir, "class_distribution_train")
val_class_distribution = plot_utils.load_class_distribution(load_dir, "class_distribution_val")


# Specify the directory to save the plots
save_dir = "Homework10/visualisation/plots/analysis"


# Plot the class distribution data for training and validation
plot_class_distribution(train_class_distribution, val_class_distribution, save_dir, image_name='Class Distribution Of Training and Validation Data.png')

