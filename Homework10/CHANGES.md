# CHANGES.md

## Project Changes To The Milestone "Project Report and Visualization" (23.05.2024)

1. **CNN_utils usability enhancement**: 
1.1 move_img() method now asks the user whether to change the destination path or to overwrite the destination folder/image if already existing to increase usability.
1.1 move_img() method now allows to keep the original image/path with the parameter keep_original=True.

2. **Img_Gallery_Examples alignment**:
2.1 Updated with latest move_img() version.

3. **Img_Gallery usability enhancement**:
3.1 Two helper functions added to ask the user for the number of plots and for patient status to plot.
3.2 Improvement in the plot_ct_scans method for nicer plots

4. **Added user input in CNN.py**:
4.1 User is now able to select the model (mlp/cnn) and task (train/regularization).
4.2 Regularization methods (early stopping/l2/data augmentation)

## Project Changes To The Milestone "Implementation And Testing Of The CNN" (16.05.2024)

### Changes Made
1. **Project scope**: It was adapted from training the nnUnet to a CNN. The focus will rely on building the components of the image gallery editor and additionally some CNN applications for further analysis.
    1.1 Converted .ipynb files to .py
    1.2 Converted all nnUNet or UNet application to strictly use CNN Deep learning models.
2. **Refactored Long Methods**: Existing functions for editing and pre-processing images were moved to a class called ImgEditorGallery to the file Img_Gallery.py.
    2.1 Using Pylint, potential issues were fixed.  
    2.2 The method move_img() was adapted
3. **Code Readability**: Added more docstrings and comments.
4. **Documentation**: Updated README.md and ROADMAP.md. Created CONTRIBUTING.md.
5. **Deep Learning GPU**: Added a script for UBELIX. This was not tested yet. It will be ready for the next submission of Task 3.
6. **Requirements**: A requirements.txt was created.

### Why These Changes Were Made
1. **Project scope**: This is to comply with the requirements of the lecture given by our advisors and to revolve around the project "Image Gallery Editor" as closely as possible for grading.
2. **Refactored Long Methods**: In order to improve modularization (Decomposition), re-use, allow simplification of the code (Abstraction) and improve code readability (Readability). Also the adaptation of move_img() allows not only to move images but also entire folders with images.
3. **Code Readability**: This was primarily due to running Pylint but also to improve overall code readability.
4. **Documentation**: Due to overall change of the project and adding primarily a guide to the user on how to run our project in the terminal based on the feedback that was given to us. For future contributors, we also added a way to do this in a separate file.
5. **Deep Learning GPU**: This is to train the model for more epochs, if interested in a more detailed and regularized analysis. As mentioned, it will be ready for Task 3.
6. **Requirements**: This is to fulfill the pre-requisites for running our project in the first place.