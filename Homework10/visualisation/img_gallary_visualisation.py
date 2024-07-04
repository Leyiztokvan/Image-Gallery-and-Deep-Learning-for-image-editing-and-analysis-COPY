import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import plot_utils

def create_plot_img_original_and_result(path_original_img: str, path_result_img: str, show_plot: bool = True, original_img_title: str = "Original Image", result_img_title: str = "Resulted Image", plot_title: str = "Plot Title", fig_size: list = [12, 6], axis: str = 'on') -> plt.Figure:
    """
    Plot two images side by side in subplots.

    Parameters:
    - path_original_img (str): Path to the original image.
    - path_result_img (str): Path to the resulting image after changes.
    - show_plot (bool): Boolean to show the plot. Default is True.
    - original_img_title (str): Title for the original image subplot. Default is 'Original Image'.
    - result_img_title (str): Title for the resulted image subplot. Default is 'Resulted Image'.
    - plot_title (str): Title for the entire plot. Default is "Plot Title".
    - fig_size (list): Dimension of the figure. Default is [12, 6].
    - axis (str): Axis of the two subplots. Default is "on". If you don't want to show axis, set to 'off'.

    Returns:
    - matplotlib.pyplot.Figure: The created figure object.
    """
    # Open the images and convert them to numpy arrays
    original_img = np.array(Image.open(path_original_img))
    result_img = np.array(Image.open(path_result_img))

    # Create a new figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0], fig_size[1])) 

    # Plot the original image on the first subplot
    ax[0].set_title(original_img_title)
    ax[0].imshow(original_img, aspect='auto')
    ax[0].axis(axis)  # Hide or show axis

    # Plot the resulted image on the second subplot
    ax[1].set_title(result_img_title)
    ax[1].imshow(result_img, aspect='auto')
    ax[1].axis(axis)  # Hide or show axis

    # Add a label to the entire plot
    fig.suptitle(plot_title, fontsize=16)

    # Adjust the layout to make more space for the title
    plt.tight_layout()

    # Display the plot 
    if show_plot:
        plt.show()

    return fig

# Visualize the functionalities of the Image Gallery with some plots

# Specify the directory in which the plots will be saved
save_dir = 'Homework10/visualisation_v3/Image_Editor_result_visualisation'

# Example of changing the grayscale of an image (alpha value)
# Specify path of each image
original_img = 'Homework10/building_images/BUILDINGS.jpg'
result_img = 'Homework10/building_images/buildings2.jpg'

grayscale_img = create_plot_img_original_and_result(
    path_original_img=original_img, 
    path_result_img=result_img, 
    result_img_title='Image After Changing Grayscale', 
    fig_size=[12, 4], 
    axis='off'
)

# Example of resizing an image
pne_img = "assignment10/resized_Img/Pneumonia_Images/RELABELED_pneumonia_image_5216.jpeg"
pne_img_resized = "assignment10/resized_Img/Resized_Pneumonia/resized_pneumonia_image_5216.jpeg"

resize_img = create_plot_img_original_and_result(
    path_original_img=pne_img, 
    path_result_img=pne_img_resized, 
    result_img_title='Resized Image', 
    axis='on'
)

# Save plots
plot_utils.save_plot(fig=grayscale_img, img_name='changed_grayscale.png', save_dir=save_dir)
plot_utils.save_plot(fig=resize_img, img_name='resized_image.png', save_dir=save_dir)
