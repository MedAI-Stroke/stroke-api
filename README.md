# Stroke Analysis API Documentation

## Overview
This API provides endpoints for stroke prediction analysis using different modalities: facial images, arm movement data, and speech audio. Each endpoint accepts specific file types and returns standardized prediction results.

## Base URL
```
https://medaistroke.duckdns.org/api
```

## Endpoints

### 1. Face Analysis
Analyzes facial images to predict stroke probability.

**Endpoint:** `/face`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

#### Request
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | Image file containing a face |

#### Response
```json
{
    "message": "Face analysis completed",
    "result": {
        "stroke": 0,        // 0: No stroke, 1: Stroke
        "score": 0.873      // Probability score (0-1)
    }
}
```

### 2. Arm Movement Analysis
Analyzes arm movement data to predict stroke probability.

**Endpoint:** `/arm`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

#### Request
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| csv | File | Yes | CSV file containing arm movement data |

#### Response
```json
{
    "message": "Arm analysis completed",
    "result": {
        "stroke": 0,        // 0: No stroke, 1: Stroke
        "score": 0.873      // Probability score (0-1)
    }
}
```

### 3. Speech Analysis
Analyzes speech audio to predict stroke probability.

**Endpoint:** `/speech`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

#### Request
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio | File | Yes | Audio file containing speech |

#### Response
```json
{
    "message": "Speech analysis completed",
    "result": {
        "stroke": 0,        // 0: No stroke, 1: Stroke
        "score": 0.873      // Probability score (0-1)
    }
}
```

## Error Responses

### Bad Request (400)
Returned when the request is invalid (e.g., missing file, invalid file format)
```json
{
    "error": "No image file"
}
```

### Internal Server Error (500)
Returned when an unexpected error occurs during processing
```json
{
    "error": "Internal Server Error",
    "message": "Error description"
}
```

## Dependencies
- Python 11
- Flask 3.0.3
- TensorFlow 2.17.0
- OpenCV 4.10.0
- scikit-learn 1.5.2
- Additional dependencies listed in requirements.txt

## Running the API
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure all model files are present in the configured `TRAINED_MODELS_DIR`:
   - face_model.pkl
   - arm_model.pkl
   - speech_model.keras

3. Start the Flask server:
```bash
python app.py
```

## Notes
- All prediction endpoints return a standardized response format with a stroke prediction (0 or 1) and a confidence score
- In production mode, detailed error messages are suppressed for security
- Debug mode can be enabled for detailed error tracking during development