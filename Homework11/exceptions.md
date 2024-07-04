# Dealing with Exceptions

Exceptions can be dealt with try and except blocks best since it gives a lot of option on what to do with the exceptions.

For example in the "Img_Gallery" class, the "rename_img" method uses the try and except block to if the new file name already exists in a file. So the os tries to rename the file but 
then realizes it already exists and instead of raising an "FileExistsError" and stopping the application, in an except statement we handle this exception accordingly by replacing the file with the same name with then new file.
