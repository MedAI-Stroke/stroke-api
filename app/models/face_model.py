import os
import joblib
from app.preprocessing import preprocess_image
from config import TRAINED_MODELS_DIR


class FaceModel:
    def __init__(self):
        self.model = joblib.load(os.path.join(TRAINED_MODELS_DIR, 'face_model.pkl'))

    def predict(self, image_file):
        face_data = preprocess_image(image_file)
        probs = self.model.predict_proba(face_data)[0]
        fail_prob = probs[0]
        pass_prob = probs[1]

        result = (pass_prob > fail_prob).astype(int)
        
        return {"stroke": result,
                'score': pass_prob}

    