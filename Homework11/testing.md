# Unit Tests

We focused on testing the "Img_Gallery" class and some of its functions since it is required a lot in this project and works as a foundation so that the rest o fthe application works.

## Test cases

Before we get to the actual tests we used "setUp" and "tearDown" methods in the unittest framework so we could prepare and simulate some images and files to help test the functions better.

**1. Test the extraction of images with an invalid file.**

It is an unexpected behaviour to put unsupported file types as the input and should raise a warning that instructs which formats are supported.

**2. Test the selection of images**
- Especially with irregular names and image types it should return no images that are selected

**3. Test opening the image by assuring the instance is the image object**

**4. Test moving the images**

**5. Test renaming the images**

**6. Test renaming the images with an empty name**
- Putting an empty name could produce unexpected behaviour since the os needs a name so it can rename the file but since we implemented an exception handler it catches it and warns the user that the new name can not be empty.

