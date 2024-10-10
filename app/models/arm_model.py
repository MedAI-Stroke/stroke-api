import os
import pickle
from app.preprocessing import preprocess_csv

from config import TRAINED_MODELS_DIR

class ArmModel:
    def __init__(self):
        model_path = os.path.join(TRAINED_MODELS_DIR, 'arm_model.pkl' )
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        
    def predict(self, csv_file):
        df = preprocess_csv(csv_file)
        pred_cls = self.model.predict(df)
        proba = self.model.predict_proba(df)[:,1][0]

        return {"stroke": pred_cls,
                "score":proba} 


