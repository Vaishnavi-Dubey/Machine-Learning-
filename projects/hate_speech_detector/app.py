from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model_path = 'model.joblib'
vectorizer_path = 'vectorizer.joblib'

# Initialize as None, we'll load them from the notebook later
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return True
    return False

def predict_hate_speech(text):
    if model is None or vectorizer is None:
        return {"error": "Model not loaded"}
    
    # Transform the text
    text_vectorized = vectorizer.transform([text])
    
    # Get prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Map prediction to label
    label_map = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither"
    }
    
    return {
        "text": text,
        "prediction": int(prediction),
        "label": label_map[prediction],
        "confidence": float(max(model.predict_proba(text_vectorized)[0]))
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"})
        
        result = predict_hate_speech(text)
        return jsonify(result)

if __name__ == '__main__':
    if load_model():
        app.run(debug=True)
    else:
        print("Error: Model files not found. Please train and save the model first.") 