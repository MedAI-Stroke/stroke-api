import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(os.path.abspath(__file__))

# from pydub import AudioSegment

# def check_audio_extension(filename):
#     _, file_extension = os.path.splittext(filename)
#     if file_extension.lower() != '.wav':
#         try:
#             audio = AudioSegment.from_file()

def preprocess_audio(audio_file):
    sr = 16000
    y, _ = librosa.load(audio_file, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_normalized = y_trimmed/ np.max(np.abs(y_trimmed))

    # MFCC Extraction
    mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr)

    # Fixed Audio Length
    current_length = mfcc.shape[1]
    target_length = 236
    if current_length > target_length:
        # Truncation
        start = (current_length - target_length) // 2
        fixed_mfcc= mfcc[:, start:start+target_length]
    elif current_length < target_length:
        # Padding
        pad_before = (target_length - current_length) // 2
        pad_after = target_length - current_length - pad_before
        fixed_mfcc = np.pad(mfcc, ((0, 0), (int(pad_before), int(pad_after))), mode='constant')


    # Standardization
    mean = np.load(os.path.join(base_dir, 'paramaters', 'audio_mean_train.npy'), allow_pickle=True)
    std = np.load(os.path.join(base_dir, 'parameters', 'audio_std_train.npy'), allow_pickle=True)
    mfcc = (fixed_mfcc-mean) / (std + 1e-8)

    # (20, 236) -> (1, 20, 236)
    return np.expand_dims(mfcc, axis=0)
