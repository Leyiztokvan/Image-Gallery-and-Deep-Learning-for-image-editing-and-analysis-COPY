from Homework10.Img_Gallery import *
import unittest
import tempfile
import os
import shutil

class TestImgGallery(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.img_editor = ImgEditorGallery(default_path=self.test_dir)

        # Create some test images
        self.test_images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
            img_path = os.path.join(self.test_dir, f'test_image_{i}.png')
            img.save(img_path)
            self.test_images.append(img_path)

        # Create an invalid file for testing the extraction method
        self.invalid_file_path = os.path.join(self.test_dir, 'invalid_file.txt')
        with open(self.invalid_file_path, 'w') as f:
            f.write("This is an invalid file for image extraction testing.")

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_extract_images_invalid_file(self):
        with self.assertRaises(ValueError) as context:
            self.img_editor.extract_images(self.invalid_file_path)
        self.assertEqual(str(context.exception), "Unsupported directory format. Only zip, tar files, or directories are supported.")

    def test_select_images(self):
        selected_images = self.img_editor.select_images(self.test_dir, image_type='png')
        self.assertEqual(len(selected_images), 5, "Should have 5 images.")

        selected_images = self.img_editor.select_images(self.test_dir, image_type='jpeg')
        self.assertEqual(len(selected_images), 0, "Should bhave no images because we dont expect the '.'")

        selected_images = self.img_editor.select_images(self.test_dir, image_name='test_image_1')
        self.assertEqual(len(selected_images), 1, "Should have 1 image.")
        self.assertIn('test_image_1.png', selected_images[0])

        selected_images = self.img_editor.select_images(self.test_dir, image_name="0000crmc'irmci")
        self.assertEqual(len(selected_images), 0, "should have no images.")

    def test_open_img(self):
        img = self.img_editor.open_img(self.test_images[0])
        self.assertIsInstance(img, Image.Image)

    '''
    def test_resize_img(self):
        img = self.img_editor.resize_img(self.test_images[0], 50, 50)
        self.assertEqual(img.size, (50, 50))
    
     def test_resize_img_invalid_size(self):
        img_path = self.test_images[0]
        with self.assertRaises(ValueError) as context:
            self.img_editor.resize_img(img_path, -100, 100)
        self.assertEqual(str(context.exception), "Width must be a positive integer.")

        with self.assertRaises(ValueError) as context:
            self.img_editor.resize_img(img_path, 100, -100)
        self.assertEqual(str(context.exception), "Height must be a positive integer.")
    '''

    def test_move_img(self):
        new_folder = os.path.join(self.test_dir, 'new_folder')
        self.img_editor.move_img(self.test_images[0], new_folder)
        new_img_path = os.path.join(new_folder, os.path.basename(self.test_images[0]))
        self.assertTrue(os.path.isfile(new_img_path))
        self.assertFalse(os.path.isfile(self.test_images[0]))

    def test_rename_img(self):
        new_img_path = os.path.join(self.test_dir, 'renamed_image.png')
        self.img_editor.rename_img(self.test_images[0], new_img_path)
        self.assertTrue(os.path.isfile(new_img_path))
        self.assertFalse(os.path.isfile(self.test_images[0]))

        # Test for no name
        new_img_path = os.path.join(self.test_dir, 'test_image.png')
        with self.assertRaises(ValueError) as context:
            self.img_editor.rename_img(new_img_path, "")
        self.assertEqual(str(context.exception), "Image path and new image path cannot be empty.")

    def test_rename_img_empty_name(self):
        new_img_path = os.path.join(self.test_dir, 'test_image.png')
        with self.assertRaises(ValueError) as context:
            self.img_editor.rename_img(new_img_path, "")
        self.assertEqual(str(context.exception), "Image path and new image path cannot be empty.")

if __name__ == '__main__':
    unittest.main()
