import os
import zipfile
import tarfile
import shutil
import math
from random import sample
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
from torchvision.transforms import ToPILImage

class ImgEditorGallery:
    def __init__(self, default_path=None):
        """
        Generate class object for managing images. 
        It will extract to specified path or to current path by default
        """
        self.default_path = default_path or os.getcwd()
        

    def extract_images(self, directory_path, extract_to=None):
        """
        Extract all images from a directory if the directory is compressed with tar or zip
        If the compressed directory is not supported raise an error.
        Else if the directory is not compressed then return the directory path

        Args:
            directory_path (str): Path to the directory.
            extract_to (str): Directory to extract the dataset to.

        Returns:
            str: Path to the extracted dataset directory.
        """
        extract_to = extract_to or self.default_path
        if os.path.isdir(directory_path):
            dataset_dir = directory_path
        elif directory_path.endswith('.zip'):
            with zipfile.ZipFile(directory_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            dataset_dir = os.path.join(extract_to, os.path.splitext(os.path.basename(directory_path))[0])
        elif directory_path.endswith('.tar') or directory_path.endswith('.tar.gz'):
            with tarfile.open(directory_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
            dataset_dir = os.path.join(extract_to, os.path.splitext(os.path.basename(directory_path))[0])
        else:
            raise ValueError("Unsupported directory format. Only zip, tar files, or directories are supported.")      
        return dataset_dir
    


    def select_images(self, directory, image_name=None, image_type=None, min_resolution=None, max_filesize=None):
        """
        Selects images from the directory based on specified criteria.

        Args:
            directory (str): Path to the directory
            image_name (str): Name of the image file to select.
            image_type (str): Type of the image file (e.g., 'jpg', 'png').
            min_resolution (tuple): Minimum resolution of the images (width, height).
            max_filesize (int): Maximum filesize of the images in bytes.

        Returns:
            list: List of paths to selected image files.
        """       
        selected_images = []
        # os.walk() traverses directory tree top-down or bottom-up. Produces a 3-tuple (root path, sub-directories, filenames)
        for root, _, files in os.walk(directory):
            for file in files:
                
                # Split filepath into name and extension to allow for more trivial search inputs 
                file_name, file_extension = os.path.splitext(file)               
                
                # Check if the desired name is contained in the file name 
                # so we can bulk select all images with the name in it. 
                # For example if the files for them are named differently by number like: buildings1, buildings2, buildings3, ...
                # Double condition: 1. check if imag_name is not 'None' 2. Substring check in file_name
                if image_name and image_name not in file_name:
                    continue
                if image_type and image_type not in file_extension[1:]:
                    continue               
                file_path = os.path.join(root, file)
                # Select by min_resolution or max_filesize
                if min_resolution or max_filesize:
                    img = Image.open(file_path)
                    if min_resolution and (img.size[0] < min_resolution[0] or img.size[1] < min_resolution[1]):
                        continue
                    if max_filesize and os.path.getsize(file_path) > max_filesize:
                        continue
                    selected_images.append(file_path)
                else:
                    selected_images.append(file_path)       
        return selected_images



    def open_img(self, filename, grayscale=None):
        """
        Open an image file using PIL.

        Parameters:
        - filename (str): The path to the image file.
        - grayscale (int): The value of the grayscale. 

        Returns:
        - PIL.Image.Image: The opened image object.

        Note:
        There is no exception handling in this function because it is handled by the .open() function internally.
        """    
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")   
        
        img = Image.open(filename)
        img.load()
        if grayscale is True:
            img = img.convert("L", colors=grayscale) # palette=Image.ADAPTIVE not used        
        return img



    def resize_img(self, image_path, width, height):
        """
        Resize an image to a given width and height.
        
        Parameters:
        - image_path (str): The path to the image file.
        - width (int): The desired width of the resized image.
        - height (int): The desired height of the resized image.

        Returns:
        - PIL.Image.Image: The resized image object.

        Raises:
        - ValueError: If the width or height is not a positive integer.
        """
        _, filename = os.path.split(image_path)
        img = self.open_img(image_path)
        original_width, original_height = img.size
        
        
        print(f"Lets resize {filename}")
        print(f"Original image size is {original_width}x{original_height}")
        
        while True:
            try:
                width = int(input("Enter the desired width: "))
                if width <= 0:
                    raise ValueError("Width must be a positive integer.")
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for width.")
        
        while True:
            try:
                height = int(input("Enter the desired height: "))
                if height <= 0:
                    raise ValueError("Height must be a positive integer.")
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for height.")
                
        img = self.open_img(image_path)
        return img.resize((width, height))



    def move_img(self, image_path, new_folder_path=None, keep_original=False):
        """
        Move an image or folder to a new or another folder.
        If no new_folder_path provided it will by default move it to the current folder.

        Parameters:
        - image_path (str): The path to the folder or image file.
        - new_folder_path (str): The path to the destination folder.
        - keep_original: Allows to keep the original image. By default false.

        Raises:
        - ValueError: If the image path or new folder path is empty or if the image is not a valid image file.
        - NotADirectoryError: If the provided new folder path is not a directory.
        """       
        new_folder_path = new_folder_path or self.default_path   
        if not image_path or not new_folder_path:
            raise ValueError("Image path and new folder path cannot be empty")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        # Case where the folder already exists
        if os.path.exists(new_folder_path):
            print(f"Folder {new_folder_path} already exists. It will be replaced")
            shutil.rmtree(new_folder_path)
            os.makedirs(new_folder_path)  
                  
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        
        # Case where image_path is a folder   
        if os.path.isdir(image_path):
            if not os.path.isdir(new_folder_path):
                raise NotADirectoryError("The provided path is not a directory")
            shutil.move(image_path, new_folder_path) 
            #for item in os.listdir(image_path):
            #   item_path = os.path.join(image_path, item)
            
        else:
            _, filename = os.path.split(image_path)
            img = self.open_img(image_path)
            
            if not isinstance(img, Image.Image):
                raise ValueError("Invalid image!")
            
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            if not os.path.isdir(new_folder_path):
                raise NotADirectoryError("The provided path is not a directory")
            
            new_path = os.path.join(new_folder_path, filename)
            img.save(new_path)
            
            if not keep_original:
                os.remove(image_path)
        
        

    def rename_img(self, img_fullpath, new_img_path):
        """
        Rename an image file.

        Parameters:
        - img_fullpath (str): The full path to the original image file.
        - new_img_path (str): The full path to the new image file with the desired name.

        Raises:
        - ValueError: If the image path or new image path is empty.
        """
        if not img_fullpath or not new_img_path:
            raise ValueError("Image path and new image path cannot be empty.")
        try:        
            os.rename(img_fullpath, new_img_path)
        except FileExistsError:
            print("File already exists")
            print("Replace existing file")
            os.remove(img_fullpath)
            os.rename(img_fullpath, new_img_path)
            
   
               
        
    def plot_ct_scans(self, dataset, n_images=1, label=1, gray=False):
        """
    Method to plot images from  plot_CT_scans
    Input parameter:
            dataset    - object which is accessible like ImageFolder
                         (https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder)
                         either Object[torch.Tensor][label] or Object[PIL.Image.Image][label]
            n_images   - number of random images which will be displayed from the dataset
            label      - Selecton whether samples of healthy patients (label=0) 
                         or samples of patients with Penumonia (label=1) shall be displayed
            gray       - Bool value whether images shall be displayed in grayscale or not (alternative RGB)  

    Return: 
            none       - This functions allows to plot the CT scans in this specific setting
        """
        print("\nLets plot some ct scans!")
        
        label = self._select_label()
        n_images = self._n_images()
        
        if isinstance(dataset[0][0], torch.Tensor):
            print("\nInput format: Tensor --> convert to PIL image before plotting\n")
            to_pil_transformer = ToPILImage()
        elif isinstance(dataset[0][0], PIL.Image.Image):
            print("\nAlready PIL image format --> Plot\n")
        else:
            raise TypeError("Unsupported dataset format. Supported are torch.Tensor or PIL.Image.Image\n ensure dataset can be accessed with format dataset[x][y].")

        condition = label == 0 or label == 1
        if not condition:
            raise ValueError("Something went wrong with the label - must be\n0 - healthy\n1 - pneumonia")
        filtered_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
        selected_indices = sample(filtered_indices, min(n_images, len(filtered_indices)))
        if n_images < 1 or n_images > 6:
            raise ValueError("Number of images to plot out of range - must be between 1 - 6")
        
        # Calculate rows and columns for the subplots
        n_cols = min(3, n_images)
        n_rows = math.ceil(n_images / n_cols)
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 4))
        axes = axes.flatten() if n_images > 1 else [axes]

        for idx, ax in zip(selected_indices, axes):
            if isinstance(dataset[0][0], torch.Tensor):
                img = to_pil_transformer(dataset[idx][0])
            else:
                img = dataset[idx][0]
            if gray:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.axis("off")
        
        # Hide any unused subplots
        for ax in axes[len(selected_indices):]:
            ax.axis("off")
            
        plt.tight_layout()
        plt.show()


    def _select_label(self):
        """
        Helper method to select the label for plotting.
        """
        while True:
            try:
                choice = int(input("Select samples of:\n1. Healthy patients\n2. Patients with pneumonia\nEnter choice (1 or 2): "))
                if choice == 1:
                    return 0  # Label for healthy patients
                elif choice == 2:
                    return 1  # Label for patients with pneumonia
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number (1 or 2).")
                
    def _n_images(self):
        """
        Helper method to select the number of plots
        """
        while True:
            try:
                choice = int(input("Select number of samples to be plotted. \Enter choice within range 1-6:"))
                if (7 > choice) & (choice > 0):
                    return choice
                else:
                    print("Invalic choice - must be in range 1-6")
            except ValueError:
                print("Invalid input. Please enter number in range 1-6")
                