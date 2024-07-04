# Imports
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from Img_Gallery import ImgEditorGallery 


# ### Unpack x-ray images and display
# Initialize the ImgEditorGallery class
img_editor = ImgEditorGallery()

# Extract x_ray_images. With extract_to=None it will extract it by default to the current directory
img_editor.extract_images(directory_path="x_ray_images.zip", extract_to=None)

# Move images to new folder
img_editor.move_img(image_path="chest_xray_64", new_folder_path="xray_dataset")

# Path to the dataset
data_dir = "xray_dataset/chest_xray_64"

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir)


# plot images
# Label: 1=Penumonia 0=Healthy
try:
    img_editor.plot_ct_scans(dataset, n_images=6, label=1, gray=True)
except Exception as e:
    print("Error:", e)


# Create new folder for image, resize image, safe the resized image, and then show it
# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images for uniformity
    transforms.ToTensor()  # Convert images to tensor format
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Ensure directories exist
os.makedirs("resized_Img/Pneumonia_Images", exist_ok=True)
os.makedirs("resized_Img/Resized_Pneumonia", exist_ok=True)

for i, (image_tensor, label) in enumerate(dataset):
    if label == 1:  # PNEUMONIA
        image_path = f"resized_Img/pneumonia_image_{i}.jpeg"
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert tensor to PIL Image
        pil_image.save(image_path)  # Save Image
        print(f"Saved pneumonia image {i} to {image_path}")
        
        # Resize the image and save the resized version
        resized_img = img_editor.resize_img(image_path, 125, 125)
        resized_img_path = f"resized_Img/Resized_Pneumonia/resized_pneumonia_image_{i}.jpeg"
        resized_img.save(resized_img_path)
        
        # Move the original image to a new directory
        new_folder_path = "resized_Img/Pneumonia_Images"
        new_image_path = f"{new_folder_path}/pneumonia_image_{i}.jpeg"
        
        img_editor.move_img(image_path, new_folder_path, keep_original=True)
        #except: print(f"{new_folder_path}/pneumonia_image_{i}.jpeg already moved to {new_folder_path}")
        
        
        # try:        
        #     os.rename(new_image_path, new_folder_path)
        # except FileExistsError:
        #     print("File already exists")
        #     print("Replace existing file")
        #     os.remove(new_image_path)
        #     os.rename(new_image_path, new_folder_path)

        
        if i >= 5:  # Limiting the number of images processed
            break

# Plot resized image using matplotlib
original_img = Image.open(new_image_path)   
resized_wave_img = Image.open(resized_img_path)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[1].imshow(resized_wave_img)
ax[1].set_title("Resized Image")
plt.show()

# Rename an image
new_img_name = f"resized_Img/Pneumonia_Images/RELABELED_pneumonia_image_{i}.jpeg"
img_editor.rename_img(new_image_path, new_img_name)


# ## Examples using all functions together to extract images from a compressed folder, open specific images in greyscale, resize and rename one image and moving it to another folder.


images = "building_images.zip"

# Extract the images and select specific all building images to open
extracted_dir = img_editor.extract_images(images)
buildings = img_editor.select_images(extracted_dir, image_name="buildings")

# Open the selected images in greyscale
for image in buildings:
    greyscaled_img = img_editor.open_img(image, 10)
    greyscaled_img.show()


# Select the forest image and move it to a new folder
forest_images = img_editor.select_images(extracted_dir, image_name="forest")

for forest_img in forest_images: # In a loop since the selected image paths are in a list and we could also bulk move if we selected multiple forest images we want to move to a new folder
    img_editor.move_img(forest_img, "Forests")



# Example of resizing an image 
img_path = "building_images/buildings.jpg"
original_img = img_editor.open_img(img_path)
resized_wave_img = img_editor.resize_img(img_path, 125, 125)

print("orginal image: ", original_img.size, original_img.mode, original_img.format)
print("resized image: ", resized_wave_img.size, resized_wave_img.mode, resized_wave_img.format)

# Plot images with matplotlib
# Display images using Matplotlib
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_img)
ax[0].set_title("Original Image")
ax[1].imshow(resized_wave_img)
ax[1].set_title("Resized Image")
plt.show()

# Rename Image
image_path = "building_images/buildings.jpg"
new_img_name = "building_images/BUILDINGS.jpg"

img_editor.rename_img(image_path, new_img_name)
img_editor.open_img(filename="building_images/BUILDINGS.jpg")


