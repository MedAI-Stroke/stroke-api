import os
import unittest
import pandas as pd
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
        processed_data = preprocess_csv(self.positive_csv)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.shape[1], 11)  # Expected: 11 PCA components

    def test_model_consistency(self):
        # Test model consistency: same input should always return the same output
        result1 = self.arm_model.predict(self.positive_csv)
        result2 = self.arm_model.predict(self.positive_csv)
        self.assertEqual(result1, result2)

    def test_invalid_csv_format(self):
        # Test handling of invalid CSV format
        invalid_csv = os.path.join(TEST_EXAMPLES_DIR, 'invalid_sample.csv')
        with open(invalid_csv, 'w') as f:
            f.write("Invalid,CSV,Format\n1,2,3\n")
        
        with self.assertRaises(Exception):
            self.arm_model.predict(invalid_csv)

if __name__ == '__main__':
    unittest.main()