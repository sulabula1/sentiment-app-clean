from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# === Ścieżki ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
TFIDF_MODEL_PATH = os.path.join(ROOT_DIR, "models", "tfidf_logreg.joblib")
HERBERT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "herbert_sentiment")

# === Aplikacja FastAPI ===
app = FastAPI(title="Analiza Sentymentu - TF-IDF & HerBERT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model wejściowy ===
class Opinion(BaseModel):
    text: str

# === Lazy loading modeli ===
_tfidf = None
_herbert = None
_tokenizer = None

def get_tfidf():
    global _tfidf
    if _tfidf is None:
        if not os.path.exists(TFIDF_MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model TF-IDF nie został znaleziony.")
        print("Wczytywanie modelu TF-IDF...")
        _tfidf = joblib.load(TFIDF_MODEL_PATH)
        print("Model TF-IDF załadowany.")
    return _tfidf

def get_herbert():
    global _herbert, _tokenizer
    if _herbert is None or _tokenizer is None:
        if not os.path.exists(HERBERT_MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model HerBERT nie został znaleziony.")
        print("Wczytywanie modelu HerBERT...")
        _tokenizer = AutoTokenizer.from_pretrained(HERBERT_MODEL_PATH)
        _herbert = AutoModelForSequenceClassification.from_pretrained(HERBERT_MODEL_PATH)
        print("Model HerBERT załadowany.")
    return _herbert, _tokenizer

# === Testowy endpoint ===
@app.get("/")
def root():
    return {"status": "ok", "available_models": ["tfidf", "herbert"]}

# === Główny endpoint analizy ===
@app.post("/analyze")
def analyze(opinion: Opinion, model: str = Query("tfidf", enum=["tfidf", "herbert"])):
    text = opinion.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Pusty tekst.")

    # === TF-IDF ===
    if model == "tfidf":
        pipe = get_tfidf()
        pred = pipe.predict([text])[0]
        clf = pipe.named_steps["clf"]
        tfidf = pipe.named_steps["tfidf"]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(tfidf.transform([text]))[0]
            labels = clf.classes_
            proba = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}
            confidence = round(float(max(probs)), 3)
        else:
            proba = None
            confidence = None

        return {
            "model": "tfidf",
            "sentiment": pred,
            "confidence": confidence,
            "probs": proba
        }

    # === HerBERT ===
    elif model == "herbert":
        model_, tokenizer = get_herbert()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model_(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        labels = list(model_.config.id2label.values())
        pred = labels[int(np.argmax(probs))]
        proba = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}
        confidence = round(float(max(probs)), 3)
        return {
            "model": "herbert",
            "sentiment": pred,
            "confidence": confidence,
            "probs": proba
        }
