import os
from io import BytesIO
import unittest
import pandas as pd
from werkzeug.datastructures import FileStorage
from app.preprocessing import preprocess_csv
from app.models import ArmModel
from config import TEST_EXAMPLES_DIR

class TestArmModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test CSV file paths
        cls.positive_csv = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_arm.csv')
        cls.negative_csv = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_arm.csv')
        
        # Create ArmModel instance
        cls.arm_model = ArmModel()
    
    def create_file_storage(self, file_path):
        with open(file_path, 'rb') as file:
            file_content = file.read()
        return FileStorage(
            stream = BytesIO(file_content),
            filename = os.path.basename(file_path),
            content_type = 'text/csv'

        )

    def test_positive_arm_sample(self):
        # Test positive sample
        file_storage = self.create_file_storage(self.positive_csv)
        result = self.arm_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # Expected: positive
        self.assertGreater(result['score'], 0.5)  # Expected: score greater than 0.5

    def test_negative_arm_sample(self):
        # Test negative sample
        file_storage = self.create_file_storage(self.negative_csv)
        result = self.arm_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # Expected: negative
        self.assertLess(result['score'], 0.5)  # Expected: score less than 0.5

    def test_preprocessing_output_shape(self):
        # Test the shape of processed CSV
        file_storage = self.create_file_storage(self.positive_csv)
        processed_data = preprocess_csv(file_storage)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.shape[1], 11)  # Expected: 11 PCA components

    def test_model_consistency(self):
        # Test model consistency: same input should always return the same output
        file_storage1 = self.create_file_storage(self.positive_csv)
        file_storage2 = self.create_file_storage(self.positive_csv)

        result1 = self.arm_model.predict(file_storage1)
        result2 = self.arm_model.predict(file_storage2)
        self.assertEqual(result1, result2)

    def test_invalid_csv_format(self):
        # Test handling of invalid CSV format
        invalid_csv = os.path.join(TEST_EXAMPLES_DIR, 'invalid_sample.csv')
        with open(invalid_csv, 'w') as f:
            f.write("Invalid,CSV,Format\n1,2,3\n")
        
        file_storage = self.create_file_storage(invalid_csv)
        with self.assertRaises(Exception):
            self.arm_model.predict(file_storage)

if __name__ == '__main__':
    unittest.main()