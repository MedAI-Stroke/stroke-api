import os
import pickle
from app.preprocessing import csv_processing

base_dir = os.path.dirname(os.path.abspath(__file__))

class ArmModel:
    def __init__(self):
        model_path = os.path.join(base_dir, src, 'arm_model.pkl' )
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        
    def predict(self, data):
        # 실제 예측 로직을 여기에 구현
        return {"stroke": 1}  # 임시 결과