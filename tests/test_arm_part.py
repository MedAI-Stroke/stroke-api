import os
import unittest
import pandas as pd
from app.preprocessing import process_csv
from app.models import ArmModel

class TestArmModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test CSV file paths
        cls.test_csv_dir = os.path.join(os.path.dirname(__file__), 'examples')
        cls.positive_csv = os.path.join(cls.test_csv_dir, 'positive_sample_arm.csv')
        cls.negative_csv = os.path.join(cls.test_csv_dir, 'negative_sample_arm.csv')
        
        # Create ArmModel instance
        cls.arm_model = ArmModel()

    def test_positive_arm_sample(self):
        # Test positive sample
        result = self.arm_model.predict(self.positive_csv)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # Expected: positive
        self.assertGreater(result['score'], 0.5)  # Expected: score greater than 0.5

    def test_negative_arm_sample(self):
        # Test negative sample
        result = self.arm_model.predict(self.negative_csv)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # Expected: negative
        self.assertLess(result['score'], 0.5)  # Expected: score less than 0.5

    def test_preprocessing_output_shape(self):
        # Test the shape of processed CSV
        processed_data = process_csv(self.positive_csv)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.shape[1], 11)  # Expected: 11 PCA components

    def test_model_consistency(self):
        # Test model consistency: same input should always return the same output
        result1 = self.arm_model.predict(self.positive_csv)
        result2 = self.arm_model.predict(self.positive_csv)
        self.assertEqual(result1, result2)

    def test_invalid_csv_format(self):
        # Test handling of invalid CSV format
        invalid_csv = os.path.join(self.test_csv_dir, 'invalid_sample.csv')
        with open(invalid_csv, 'w') as f:
            f.write("Invalid,CSV,Format\n1,2,3\n")
        
        with self.assertRaises(Exception):
            self.arm_model.predict(invalid_csv)

if __name__ == '__main__':
    unittest.main()