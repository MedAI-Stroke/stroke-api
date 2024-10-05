from flask import abort

def validate_face_data(data):
    if not isinstance(data, dict):
        abort(400, description="Invalid data format. Expected a JSON object.")
    
    if 'image' not in data:
        abort(400, description="Missing 'image' field in the request data.")
    
    # 여기에 이미지 데이터의 형식, 크기 등을 검사하는 로직을 추가할 수 있습니다.
    # 예: base64 인코딩된 문자열인지, 최대 크기를 초과하지 않았는지 등

    return True

def validate_arm_data(data):
    if not isinstance(data, dict):
        abort(400, description="Invalid data format. Expected a JSON object.")
    
    if 'movement_data' not in data:
        abort(400, description="Missing 'movement_data' field in the request data.")
    
    # 여기에 움직임 데이터의 형식, 길이 등을 검사하는 로직을 추가할 수 있습니다.
    # 예: 리스트의 길이, 각 요소의 형식 등

    return True

def validate_speech_data(data):
    if not isinstance(data, dict):
        abort(400, description="Invalid data format. Expected a JSON object.")
    
    if 'audio' not in data:
        abort(400, description="Missing 'audio' field in the request data.")
    
    # 여기에 오디오 데이터의 형식, 길이 등을 검사하는 로직을 추가할 수 있습니다.
    # 예: base64 인코딩된 문자열인지, 최대 길이를 초과하지 않았는지 등

    return True