import os
from io import BytesIO
import numpy as np
import librosa
from config import PREPROCESSING_PARAMS_DIR


def rms_normalize(audio, target_dB=-20):
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_dB/20)
    return audio * (target_rms/rms)


def adaptive_preemphasis(audio, sr):
    spec = np.abs(librosa.stft(audio))
    freq_ratio = np.mean(spec[spec.shape[0]//2:]) / np.mean(spec[:spec.shape[0]//2])
    
    if freq_ratio < 0.1:
        alpha = 0.97
    elif freq_ratio < 0.3:
        alpha = 0.95
    else:
        alpha = 0.90
        
    return np.append(audio[0], audio[1:] - alpha * audio[:-1]), alpha


def preprocess_audio(audio_file):
    audio_bytes = audio_file.read()

    # 1. Load audio
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    
    # 2. Standardization of Sampling Rate
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # 3. Volume Normalization
    y = rms_normalize(y)

    # 4. Pre-emphasis
    y, _ = adaptive_preemphasis(y, sr=16000)

    # 5. 묵음 제거
    intervals = librosa.effects.split(y, top_db=20)
    y = np.concatenate([y[start:end] for start, end in intervals])

    # 6. 오디오 길이 표준화
    target_length = 320000 # 20초 * 16000 sr
    if len(y) < target_length : 
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

     # 7. MFCC 추출
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, hop_length= 512)
    
    # 8. MFCC 정규화
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
            (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

    
    # (20, 236) -> (1, 20, 236)
    return np.expand_dims(mfcc, axis=0)
