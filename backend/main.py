from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from backend.models import facial_emotion, voice_emotion, chatbot

app = FastAPI()

class EmotionRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to AI-powered Emotional Coach!"}

@app.post("/detect_facial_emotion")
def detect_facial_emotion(image: UploadFile = File(...)):
    # Simulate the facial emotion detection logic
    image_path = "path/to/saved/image"  # Save the uploaded file to path
    emotion, confidence = facial_emotion.detect_emotion_from_image(image_path)
    return {"emotion": emotion, "confidence": confidence}

@app.post("/detect_voice_emotion")
def detect_voice_emotion(audio: UploadFile = File(...)):
    # Simulate the voice emotion detection logic
    audio_path = "path/to/saved/audio"  # Save the uploaded audio file
    emotion, confidence = voice_emotion.detect_emotion_from_audio(audio_path)
    return {"emotion": emotion, "confidence": confidence}

@app.post("/chatbot")
def chat_with_bot(request: EmotionRequest):
    # Simulate chatbot interaction
    response = chatbot.chatbot_response(request.text)
    return {"response": response}
from transformers import pipeline

# Load pre-trained emotion detection model (BERT or RoBERTa)
emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

@app.post("/detect_emotion")
async def detect_emotion(request: EmotionRequest):
    result = emotion_model(request.text)
    return {"emotion": result[0]['label'], "confidence": result[0]['score']}
from transformers import pipeline

# Load pre-trained emotion detection model (BERT or RoBERTa)
emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

@app.post("/detect_emotion")
async def detect_emotion(request: EmotionRequest):
    result = emotion_model(request.text)
    return {"emotion": result[0]['label'], "confidence": result[0]['score']}
