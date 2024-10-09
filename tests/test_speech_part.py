import os
import unittest
import tensorflow as tf
from app.preprocessing import preprocess_audio
from app.models import SpeechModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable all CUDA devices
tf.config.set_visible_devices([], 'GPU')


class TestSpeechModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 테스트용 오디오 파일 경로 설정
        cls.test_audio_dir = os.path.join(os.path.dirname(__file__), 'examples')
        print(cls.test_audio_dir)
        cls.positive_audio = os.path.join(cls.test_audio_dir, 'positive_sample_audio.wav')
        cls.negative_audio = os.path.join(cls.test_audio_dir, 'negative_sample_audio.wav')
        
        # SpeechModel 인스턴스 생성
        cls.speech_model = SpeechModel()

    def test_positive_audio_sample(self):
        # 양성 샘플 테스트
        result = self.speech_model.predict(self.positive_audio)
        
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 1)  # 예상: 양성
        self.assertGreater(result['score'], 0.5)  # 예상: 0.5보다 큰 점수

    def test_negative_audio_sample(self):
        # 음성 샘플 테스트
        result = self.speech_model.predict(self.negative_audio)
        self.assertIsInstance(result, dict)
        self.assertIn('stroke', result)
        self.assertIn('score', result)
        self.assertEqual(result['stroke'], 0)  # 예상: 음성
        self.assertLess(result['score'], 0.5)  # 예상: 0.5보다 작은 점수
        

    def test_preprocessing_output_shape(self):
        # 전처리된 오디오의 shape 테스트
        preprocessed_audio = preprocess_audio(self.positive_audio)
        self.assertEqual(preprocessed_audio.shape, (1, 20, 236))  # 예상 shape

    def test_model_consistency(self):
        # 모델 일관성 테스트: 같은 입력에 대해 항상 같은 출력을 반환하는지
        result1 = self.speech_model.predict(self.positive_audio)
        result2 = self.speech_model.predict(self.positive_audio)
        self.assertEqual(result1, result2)

if __name__ == '__main__':
    unittest.main()