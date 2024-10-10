import os
import unittest
import numpy as np
from app.preprocessing import preprocess_image
from app.models import FaceModel
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

    def test_positive_face_sample(self):
        # Test positive sample
        with open(self.positive_image, 'rb') as f:
            image_bytes = f.read()
        result = self.face_model.predict(image_bytes)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # Expect: positive
        self.assertGreater(result['score'], 0.5)  # Expect: score > 0.5

    def test_negative_face_sample(self):
        # Test negative sample
        with open(self.negative_image, 'rb') as f:
            image_bytes = f.read()
        result = self.face_model.predict(image_bytes)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # Expect: negative
        self.assertLess(result['score'], 0.5)  # Expect: score < 0.5

    def test_preprocessing_output_shape(self):
        # Test shape of preprocessed image
        with open(self.positive_image, 'rb') as f:
            image_bytes = f.read()
        preprocessed_image = preprocess_image(image_bytes)
        self.assertEqual(preprocessed_image.shape, (1, 8))  # Expected shape

    def test_model_consistency(self):
        # Test model consistency: same input should always return same output
        with open(self.positive_image, 'rb') as f:
            image_bytes = f.read()
        result1 = self.face_model.predict(image_bytes)
        result2 = self.face_model.predict(image_bytes)
        self.assertEqual(result1, result2)

    def test_non_face_image(self):
        # Test handling of non-face image
        with open(self.non_face_image, 'rb') as f:
            image_bytes = f.read()
        with self.assertRaises(ValueError):
            self.face_model.predict(image_bytes)

if __name__ == '__main__':
    unittest.main()