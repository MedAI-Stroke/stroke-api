import os
import tensorflow as tf
from tensorflow import keras
from app.preprocessing import preprocess_audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.set_visible_devices([], 'GPU')

base_dir = os.path.dirname(os.path.abspath(__file__))

class SpeechModel:
    def __init__(self):
        self.model = keras.models.load_model(os.path.join(base_dir, 'trained', 'speech_model.keras'))
    
    def predict(self, audio_file):
        audio = preprocess_audio(audio_file)
        pred = self.model.predict(audio)
        pred_cls = (pred > 0.5).astype(int)
        return {"stroke": pred_cls[0][0],
                "score":pred[0][0]} 
