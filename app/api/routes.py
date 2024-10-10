import traceback
from flask import Blueprint, jsonify, request
from app.models import FaceModel, ArmModel, SpeechModel

api_bp = Blueprint('api', __name__)

face_model = FaceModel()
arm_model = ArmModel()
speech_model = SpeechModel()

@api_bp.route('/face', methods=['POST'])
def face_analysis():
    try: 
        if 'image' not in request.files:
            return jsonify({'error':"No image file"}), 400
        
        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if image_file:
            result = face_model.predict(image_file)
            return jsonify({"message": "Face analysis completed", "result":result}), 200
        
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@api_bp.route('/arm', methods=['POST'])
def arm_analysis():
    try:
        if 'csv' not in request.files:
            return jsonify({'error': 'No CSV files'}), 400
        
        csv_file = request.files['csv']

        if csv_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if csv_file:
            result = arm_model.predict(csv_file)
            return jsonify({"message": "Arm analysis completed", "result": result}), 200
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500 
           
@api_bp.route('/speech', methods=['POST'])
def speech_analysis():
    try: 
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
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error.description)}), 400



@api_bp.errorhandler(Exception)
def handle_exception(error):
    # 개발 환경에서만 상세 에러 메시지 포함
    if api_bp.config['DEBUG']:
        return jsonify({
            "error": "Unexpected Error",
            "message": str(error),
            "type": type(error).__name__
        }), 500
    # 프로덕션 환경에서는 일반적인 에러 메시지만 반환
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500