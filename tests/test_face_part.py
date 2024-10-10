import os
import unittest
from io import BytesIO
from werkzeug.datastructures import FileStorage

import numpy as np

from app.models import FaceModel
from app.preprocessing import preprocess_image
from config import TEST_EXAMPLES_DIR

class TestFaceModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test image file paths
        cls.positive_image = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_face.jpg')
        cls.negative_image = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_face.jpg')
        cls.non_face_image = os.path.join(TEST_EXAMPLES_DIR, 'non_face_image.jpg')
        
        # Create FaceModel instance
        cls.face_model = FaceModel()
    
    def create_file_storage(self, file_path):
        with open(file_path, 'rb') as file:
            file_content = file.read()
        return FileStorage(
            stream = BytesIO(file_content),
            filename = os.path.basename(file_path),
            content_type = 'image/jpeg'
        )

    def test_positive_face_sample(self):
        # Test positive sample
        file_storage = self.create_file_storage(self.positive_image)
        result = self.face_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # Expect: positive
        self.assertGreater(result['score'], 0.5)  # Expect: score > 0.5

    def test_negative_face_sample(self):
        # Test negative sample
        file_storage = self.create_file_storage(self.negative_image)
        result = self.face_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # Expect: negative
        self.assertLess(result['score'], 0.5)  # Expect: score < 0.5

    def test_preprocessing_output_shape(self):
        # Test shape of preprocessed image
        file_storage = self.create_file_storage(self.positive_image)
        preprocessed_image = preprocess_image(file_storage)
        self.assertIsInstance(preprocessed_image, np.ndarray)
        self.assertEqual(preprocessed_image.shape, (1, 8))  # Expected shape

    def test_model_consistency(self):
        # Test model consistency: same input should always return same output
        file_storage1 = self.create_file_storage(self.positive_image)
        file_storage2 = self.create_file_storage(self.positive_image)
        result1 = self.face_model.predict(file_storage1)
        result2 = self.face_model.predict(file_storage2)
        self.assertEqual(result1, result2)

    def test_non_face_image(self):
        # Test handling of non-face image
        file_storage = self.create_file_storage(self.non_face_image)
        with self.assertRaises(ValueError):
            self.face_model.predict(file_storage)

if __name__ == '__main__':
    unittest.main()