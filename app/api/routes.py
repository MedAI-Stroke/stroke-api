from flask import Blueprint, jsonify, request
from app.models import FaceModel, ArmModel, SpeechModel


api_bp = Blueprint('api', __name__)

face_model = FaceModel()
arm_model = ArmModel()
speech_model = SpeechModel()

@api_bp.route('/face', methods=['POST'])
def face_analysis():
    if 'image' not in request.files:
        return jsonify({'error':"No image file"}), 400
    
    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if image_file:
        result = face_model.predict(image_file)
        return jsonify({"message": "Face analysis completed", "result":result}), 200

@api_bp.route('/arm', methods=['POST'])
def arm_analysis():
    if 'csv' not in request.files:
        return jsonify({'error': 'No CSV files'}), 400
    
    csv_file = request.files['csv']

    if csv_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if csv_file:
        result = arm_model.predict(csv_file)
        return jsonify({"message": "Arm analysis completed", "result": result}), 200

@api_bp.route('/speech', methods=['POST'])
def speech_analysis():
    if 'audio' not in request.files:
        return jsonify({'error': "No Audio file"}), 400
    audio_file = request.files['audio']

    if audio_file.filename=='':
        return jsonify({'error':'No selected file'}), 400   
    
    if audio_file:
        # # 임시 저장시
        # temp_path = save_temp_file(audio_file)
        # result = speech_model.predict(temp_path)

        # 모델 직접 전달시
        result = speech_model.predict(audio_file)
        return jsonify({"message": "Speech analysis completed", "result": result}), 200



@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error.description)}), 400