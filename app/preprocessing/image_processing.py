import os
import math

import cv2
import dlib
import joblib
import numpy as np

from config import PREPROCESSING_PARAMS_DIR

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(PREPROCESSING_PARAMS_DIR, 'shape_predictor_68_face_landmarks.dat'))
scaler = joblib.load(os.path.join(PREPROCESSING_PARAMS_DIR, 'face_scaler.pkl'))
norm = joblib.load(os.path.join(PREPROCESSING_PARAMS_DIR, 'face_norm.pkl'))

def get_landmark_list(landmarks):
    landmark_list = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_list.append((x, y))
    return landmark_list

def is_frontal_face(landmarks):
    # Left eye, right eye, nose tip
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    nose_tip = (landmarks.part(27).x, landmarks.part(27).y)

    x1, y1 = nose_tip
    x2, y2 = left_eye
    x3, y3 = right_eye

    left_to_nose = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    right_to_nose = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    # Ratio of distances between eyes and nose
    ratio = left_to_nose / right_to_nose

    return ratio

def get_sym_mouth(landmark_list):
    mouth_land = landmark_list[48:]
    ml = mouth_land

    sym_mouth = [
        (ml[0], ml[6]),
        (ml[1], ml[5]),
        (ml[2], ml[4]),
        (ml[7], ml[11]),
        (ml[8], ml[10]),
        (ml[12], ml[16]),
        (ml[13], ml[15]),
        (ml[17], ml[19])
    ]

    return sym_mouth
    
def distances_ratio_with_eye(landmark_list):
    sym_mouth = get_sym_mouth(landmark_list)
    sym_mouth = [sym_mouth[0], sym_mouth[1], sym_mouth[3], sym_mouth[5]]

    left_eye = landmark_list[39]
    right_eye = landmark_list[42]
    x1, y1 = left_eye
    x3, y3 = right_eye

    distances = []

    for left_mouth, right_mouth in sym_mouth:
        x2, y2 = left_mouth
        x4, y4 = right_mouth
        left_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        right_distance = math.sqrt((x4 - x3)**2 + (y4 - y3)**2)
        distance = left_distance / right_distance
        distance = max(distance, 1/distance)
        distances.append(distance)

    return distances


def preprocess_image(image_file):
    image_bytes = image_file.read()

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        raise ValueError("No face detected. Please provide an image with a clear frontal face.")

    face = faces[0]
    landmarks = predictor(gray, face)

    # Check if the face is frontal
    ratio = is_frontal_face(landmarks)
    if not (0.88 < ratio < 1.12):
        raise ValueError(f"Face is not frontal. Ratio: {ratio}")

    # Get landmarks and calculate distances
    landmark_list = get_landmark_list(landmarks)
    distances = distances_ratio_with_eye(landmark_list)

    # Convert to numpy array and reshape
    face_data_array = np.array(distances).reshape(1, -1)

    # Scale the data
    face_data_scaled = scaler.transform(face_data_array)
    face_data_scaled = norm.transform(face_data_scaled)

    return face_data_scaled
