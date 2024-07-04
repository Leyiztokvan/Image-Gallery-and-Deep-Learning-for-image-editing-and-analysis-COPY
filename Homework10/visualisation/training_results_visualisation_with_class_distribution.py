import matplotlib.pyplot as plt
import plot_utils_v2 as plot_utils

def create_plot_result(plot_arrays: list, show_plot: bool = True, plot_title: str = 'Plot Title', plot_labels: list = ['Plot Label X ax', 'Plot Label Y ax'], plot_arrays_labels: list =['array_1  Label', 'array_2 Label'], fig_size: list = [14, 6], axis: str = 'on') -> plt.Figure:
    """
    Plot results.

    Parameters:
    - plot_arrays (list): List of lists containing data for plot. The list should contain two lists: [data1, data2].
    - show_plot (bool): Boolean to show the plot. Default is True.
    - plot_title (str): Title for the entire plot. Default is "Plot Title".
    - plot_labels (list): Labels for x and y axes for the plot. Default is ['Plot Label X ax', 'Plot Label Y ax'].
    - plot_arrays_labels (list): Labels for each array in the plot. Default is ['array_1 Label', 'array_2 Label'].
    - fig_size (list): Dimension of the figure. Default is [14, 6].
    - axis (str): Axis of the subplots. Default is "on".

    Returns: 
    - matplotlib.pyplot.Figure object
    """

    epochs = range(1, len(plot_arrays[0]) + 1)

    # Create a new figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_size[0], fig_size[1]))

    # Plot data
    ax.plot(epochs, plot_arrays[0], label=plot_arrays_labels[0])
    ax.plot(epochs, plot_arrays[1], label=plot_arrays_labels[1])
    ax.set_title(plot_title)
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.legend()
    ax.axis(axis)

    # Display the plot 
    if show_plot:
        plt.show()

    return fig



def create_plot_result_in_two_subplots(subplots_arrays: list, show_plot: bool = True, plot_title: str = "Plot Title", subplots_titles: list = ['Subplot 1', 'Subplot 2'], subplots_labels: list =[['Subplot_1  Label X ax', 'Subplot_1 Label Y ax']], subplots_array_labels: list =[['Subplot_1_array_1  Label', 'Subplot_1_array_2 Label'], ['Subplot_2_array_1  Label', 'Subplot_2_array_2  Label']], fig_size: list = [14, 6], axis: str = 'on') -> plt.Figure:
    """
    Plot results in two subplots.

    Parameters:
    - subplots_arrays (list): List of lists containing data for subplots. Each sublist should contain two lists: [data1, data2].
    - show_plot (bool): Boolean to show the plot. Default is True.
    - plot_title (str): Title for the entire plot. Default is 'Plot Title'.
    - subplots_titles (list): Titles for each subplot. Default is ['Subplot 1', 'Subplot 2'].
    - subplots_labels (list): Labels for x and y axes for each subplot. Default is [['Subplot_1  Label X ax', 'Subplot_1 Label Y ax'], ['Subplot_2  Label X ax', 'Subplot_2 Label Y ax']].
    - fig_size (list): Dimension of the figure. Default is [14, 6].
    - axis (str): Axis of the subplots. Default is "on".

    Returns: 
    - matplotlib.pyplot.Figure object
    """

    epochs = range(1, len(subplots_arrays[0][0]) + 1)

    # Create a new figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0], fig_size[1]))

    # Plot data on the first subplot
    ax[0].plot(epochs, subplots_arrays[0][0], label=subplots_array_labels[0][0])
    ax[0].plot(epochs, subplots_arrays[0][1], label=subplots_array_labels[0][1])
    ax[0].set_title(subplots_titles[0])
    ax[0].set_xlabel(subplots_labels[0][0])
    ax[0].set_ylabel(subplots_labels[0][1])
    ax[0].legend()
    ax[0].axis(axis)

    # Plot data on the second subplot
    ax[1].plot(epochs, subplots_arrays[1][0], label=subplots_array_labels[1][0])
    ax[1].plot(epochs, subplots_arrays[1][1], label=subplots_array_labels[1][1])
    ax[1].set_title(subplots_titles[1])
    ax[1].set_xlabel(subplots_labels[1][0])
    ax[1].set_ylabel(subplots_labels[1][1])
    ax[1].legend()
    ax[1].axis(axis)

    # Add a label to the entire plot
    fig.suptitle(plot_title, fontsize=16)

    # Adjust the layout to make more space for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot 
    if show_plot:
        plt.show()

    return fig



# Load training results
load_dir_mlp = "Homework10/visualisation_v3/training_results_sample_with_class_distribution/mlp"
load_dir_cnn = "Homework10/visualisation_v3/training_results_sample_with_class_distribution/cnn"

loaded_mlp_train_losses, loaded_mlp_val_losses, loaded_mlp_train_accuracies, loaded_mlp_val_accuracies = plot_utils.load_results(load_dir_mlp)
loaded_cnn_train_losses, loaded_cnn_val_losses, loaded_cnn_train_accuracies, loaded_cnn_val_accuracies = plot_utils.load_results(load_dir_cnn)


# Define subplots data
arrays_mlp = [[loaded_mlp_train_losses, loaded_mlp_val_losses], [loaded_mlp_train_accuracies, loaded_mlp_val_accuracies]]
arrays_cnn = [[loaded_cnn_train_losses, loaded_cnn_val_losses], [loaded_cnn_train_accuracies, loaded_cnn_val_accuracies]]


# Create plots
save_dir_mlp = 'Homework10/visualisation_v3/plots/mlp'
save_dir_cnn = 'Homework10/visualisation_v3/plots/cnn'

# Plot the learning curves for MLP
mlp_learning_curves = create_plot_result_in_two_subplots(subplots_arrays=arrays_mlp, plot_title='MLP Learning Curves', subplots_titles=['Training and Validation Losses', 'Training and Validation Accuracies'], subplots_labels=[['Epochs', 'Loss'], ['Epochs', 'Accuracy']], subplots_array_labels=[['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']], axis='on')
# Plot overfitting for MLP
mlp_overfitting_curve = create_plot_result(plot_arrays=arrays_mlp[0], plot_title='MLP Overfitting', plot_labels=['Epochs', 'Loss'], plot_arrays_labels=['Training Loss', 'Validation Loss'], axis='on')

# Plot the learning curves for CNN
cnn_learning_curves = create_plot_result_in_two_subplots(subplots_arrays=arrays_cnn, plot_title='CNN Learning Curves', subplots_titles=['Training and Validation Losses', 'Training and Validation Accuracies'], subplots_labels=[['Epochs', 'Loss'], ['Epochs', 'Accuracy']], subplots_array_labels=[['Training Loss', 'Validation Loss'], ['Training Accuracy', 'Validation Accuracy']], axis='on')
# Plot overfitting for CNN
cnn_overfitting_curve = create_plot_result(plot_arrays=arrays_cnn[0], plot_title='CNN Overfitting', plot_labels=['Epochs', 'Loss'], plot_arrays_labels=['Training Loss', 'Validation Loss'], axis='on')

# Save plots
plot_utils.save_plot(fig=mlp_learning_curves, img_name='MLP Learning Curves.png', save_dir=save_dir_mlp)
plot_utils.save_plot(fig= mlp_overfitting_curve, img_name='MLP Overfitting.png', save_dir=save_dir_mlp)

plot_utils.save_plot(fig= cnn_learning_curves, img_name='CNN Learning Curves.png', save_dir=save_dir_cnn)
plot_utils.save_plot(fig= cnn_overfitting_curve, img_name='CNN Overfitting.png', save_dir=save_dir_cnn)
