from flask import Blueprint, jsonify, request
from app.models.face_model import FaceModel
from app.models.arm_model import ArmModel
from app.models.speech_model import SpeechModel

api_bp = Blueprint('api', __name__)

face_model = FaceModel()
arm_model = ArmModel()
speech_model = SpeechModel()

@api_bp.route('/face', methods=['POST'])
def face_analysis():
    # 여기에 face 모델 로직을 구현합니다
    data = request.json
    result = face_model.predict(data)
    return jsonify({"message": "Face analysis completed", "result":result}), 200

@api_bp.route('/arm', methods=['POST'])
def arm_analysis():
    # 여기에 arm 모델 로직을 구현합니다
    data = request.json
    result = arm_model.predict(data)
    return jsonify({"message": "Arm analysis completed", "result": result}), 200

@api_bp.route('/speech', methods=['POST'])
def speech_analysis():
    # 여기에 speech 모델 로직을 구현합니다
    data = request.json
    result = speech_model.predict(data)
    return jsonify({"message": "Speech analysis completed", "result": result}), 200



@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error.description)}), 400