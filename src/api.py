"""
api.py

FastAPI service exposing the duplicate-question model.
Run locally with:  uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import predict

app = FastAPI(title="Quora Duplicate Question Detector")

# Loosen this to your actual frontend origin(s) before going to production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionPair(BaseModel):
    question1: str
    question2: str


class PredictionResponse(BaseModel):
    is_duplicate: bool
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_duplicate(pair: QuestionPair):
    if not pair.question1.strip() or not pair.question2.strip():
        raise HTTPException(status_code=400, detail="Both questions must be non-empty.")

    result = predict(pair.question1, pair.question2)
    return PredictionResponse(**result)