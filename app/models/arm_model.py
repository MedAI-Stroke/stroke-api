import os
import pickle
from app.preprocessing import process_csv

base_dir = os.path.dirname(os.path.abspath(__file__))

class ArmModel:
    def __init__(self):
        model_path = os.path.join(base_dir, 'trained', 'arm_model.pkl' )
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        
    def predict(self, csv_file):
        df = process_csv(csv_file)
        pred_cls = self.model.predict(df)
        proba = self.model.predict_proba(df)[:,1][0]
        print(proba)
        print(pred_cls)

        return {"stroke": pred_cls,
                "score":proba} 


