# MedAI-Stroke API

MedAI-Stroke API는 뇌졸중 진단을 위한 얼굴 비대칭, 팔 움직임, 언어 장애를 분석하는 API입니다.

## 엔드포인트

### 1. 얼굴 비대칭 탐지

- URL: `/api/face`
- 메소드: POST
- 요청 본문:
  ```json
  {
    "image": "이미지 데이터"
  }
  ```
- 응답 예시:
  ```json
  {
  "message": "Face analysis completed",
  "result": {
    "stroke": 1
  }
    }
    ```
### 2. 운동 비대칭 탐지

- URL: `/api/arm`
- 메소드: POST
- 요청 본문:
  ```json
  {
    "csv": "센서 데이터"
  }
  ```
- 응답 예시:
  ```json
  {
  "message": "Arm analysis completed",
  "result": {
    "stroke": 1
  }
    }
    ```


### 3. 언어 장애 탐지

- URL: `/api/speech`
- 메소드: POST
- 요청 본문:
  ```json
  {
    "audio": "음성 오디오 데이터"
  }
  ```
- 응답 예시:
  ```json
  {
  "message": "Speech analysis completed",
  "result": {
    "stroke": 1,
    "score":0.885
    
  }
    }
    ```


## Android Kotlin에서의 사용 예시
```kotlin
interface MedAIStrokeApi {
    @POST("api/face")
    suspend fun analyzeFace(@Body faceData: FaceData): Response<FaceAnalysisResult>

    @POST("api/arm")
    suspend fun analyzeArm(@Body armData: ArmData): Response<ArmAnalysisResult>

    @POST("api/speech")
    suspend fun analyzeSpeech(@Body speechData: SpeechData): Response<SpeechAnalysisResult>
}

// Retrofit 인스턴스 생성
val retrofit = Retrofit.Builder()
    .baseUrl("http://13.54.38.182:8080/")
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val api = retrofit.create(MedAIStrokeApi::class.java)

// API 호출 예시
try {
    val response = api.analyzeFace(faceData)
    if (response.isSuccessful) {
        val result = response.body()
        // 결과 처리
    } else {
        // 에러 처리
    }
} catch (e: Exception) {
    // 네트워크 에러 처리
}
```