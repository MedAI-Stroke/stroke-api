import unittest
import json
from flask import Flask
from app.api.routes import api_bp
from config import TEST_EXAMPLES_DIR
import os

class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(api_bp, url_prefix='/api')
        self.client = self.app.test_client()
        
        # Test file paths
        self.positive_csv = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_arm.csv')
        self.negative_csv = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_arm.csv')
        self.positive_audio = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_audio.wav')
        self.negative_audio = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_audio.wav')
        self.positive_image = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_face.jpg')
        self.negative_image = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_face.jpg')

    def test_face_analysis_positive(self):
        with open(self.positive_image, 'rb') as image_file:
            response = self.client.post(
                '/api/face',
                data={'image': (image_file, 'positive_sample_face.jpg')},
                content_type='multipart/form-data'
            )
        

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 1)
        self.assertGreater(data['result']['score'], 0.5)

    def test_face_analysis_negative(self):
        with open(self.negative_image, 'rb') as image_file:
            response = self.client.post(
                '/api/face',
                data={'image': (image_file, 'negative_sample_face.jpg')},
                content_type='multipart/form-data'
            )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 0)
        self.assertLess(data['result']['score'], 0.5)

    def test_arm_analysis_positive(self):
        with open(self.positive_csv, 'rb') as csv_file:
            response = self.client.post(
                '/api/arm',
                data={'csv': (csv_file, 'positive_sample_arm.csv')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 1)
        self.assertGreater(data['result']['score'], 0.5)

    def test_arm_analysis_negative(self):
        with open(self.negative_csv, 'rb') as csv_file:
            response = self.client.post(
                '/api/arm',
                data={'csv': (csv_file, 'negative_sample_arm.csv')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 0)
        self.assertLess(data['result']['score'], 0.5)

    def test_speech_analysis_positive(self):
        with open(self.positive_audio, 'rb') as audio_file:
            response = self.client.post(
                '/api/speech',
                data={'audio': (audio_file, 'positive_sample_audio.wav')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 1)
        self.assertGreater(data['result']['score'], 0.5)

    def test_speech_analysis_negative(self):
        with open(self.negative_audio, 'rb') as audio_file:
            response = self.client.post(
                '/api/speech',
                data={'audio': (audio_file, 'negative_sample_audio.wav')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertEqual(data['result']['stroke'], 0)
        self.assertLess(data['result']['score'], 0.5)

    def test_missing_file(self):
        response = self.client.post('/api/face', data={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No image file')

        response = self.client.post('/api/arm', data={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No CSV files')

        response = self.client.post('/api/speech', data={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No Audio file')

if __name__ == '__main__':
    unittest.main()