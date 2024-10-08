import os
import unittest
from app.models import SpeechModel

base_dir = os.path.dirname(os.path.abspath(__file__))

class TestSpeechModel(unittest.TestCase):
    
    def setUp(self):
        self.model = SpeechModel()
        
    def test_prediction(self):
        # 여기에 오디오 파일 경로를 설정
        audio_file = os.path.join(base_dir, 'src', 'test_audio.wav')
        result = self.model.predict(audio_file)
        self.assertIn("stroke", result)  # 응답에 'stroke' 키가 있는지 확인
        self.assertIn("score", result)   # 응답에 'score' 키가 있는지 확인
        print(result)

if __name__ == '__main__':
    unittest.main()
