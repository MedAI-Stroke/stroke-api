import os
import unittest
from io import BytesIO
from werkzeug.datastructures import FileStorage
import numpy as np

from app.models import SpeechModel
from app.preprocessing import preprocess_audio
from config import TEST_EXAMPLES_DIR

class TestSpeechModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 테스트용 오디오 파일 경로 설정
        cls.positive_audio = os.path.join(TEST_EXAMPLES_DIR, 'positive_sample_audio.wav')
        cls.negative_audio = os.path.join(TEST_EXAMPLES_DIR, 'negative_sample_audio.wav')
        
        # SpeechModel 인스턴스 생성
        cls.speech_model = SpeechModel()

    
    def create_file_storage(self, file_path):
        with open(file_path, 'rb') as file:
            file_content = file.read()
        return FileStorage(
            stream=BytesIO(file_content),
            filename=os.path.basename(file_path),
            content_type='audio/wav'
        )

    def test_positive_audio_sample(self):
        # 양성 샘플 테스트
        file_storage = self.create_file_storage(self.positive_audio)
        result = self.speech_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # 예상: 양성
        self.assertGreater(result['score'], 0.5)  # 예상: 0.5보다 큰 점수

    def test_negative_audio_sample(self):
        # 음성 샘플 테스트
        file_storage = self.create_file_storage(self.negative_audio)
        result = self.speech_model.predict(file_storage)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # 예상: 음성
        self.assertLess(result['score'], 0.5)  # 예상: 0.5보다 작은 점수
        

    def test_preprocessing_output_shape(self):
        # 전처리된 오디오의 shape 테스트
        file_storage = self.create_file_storage(self.positive_audio)
        preprocessed_audio = preprocess_audio(file_storage)

        self.assertIsInstance(preprocessed_audio, np.ndarray)
        self.assertEqual(preprocessed_audio.shape, (1, 13, 626))  # 예상 shape

    def test_model_consistency(self):
        # 모델 일관성 테스트: 같은 입력에 대해 항상 같은 출력을 반환하는지
        file_storage1 = self.create_file_storage(self.positive_audio)
        file_storage2 = self.create_file_storage(self.positive_audio)
        result1 = self.speech_model.predict(file_storage1)
        result2 = self.speech_model.predict(file_storage2)
        self.assertEqual(result1, result2)

if __name__ == '__main__':
    unittest.main()