# 🚀 Machine Learning — Projects & Algorithms

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Vaishnavi-Dubey/Machine-Learning-.svg?style=for-the-badge)](https://github.com/Vaishnavi-Dubey/Machine-Learning-/stargazers)

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

> A comprehensive collection of Machine Learning projects — from classical algorithms (Decision Trees, Random Forest) to production-ready applications including **Healthcare Prediction**, **Hate Speech Detection**, **Real Estate Price Prediction**, and **Movie Sentiment Analysis**.

---

## ✨ Key Features

- 🏥 **Healthcare ML** — Heart disease and general disease prediction using clinical datasets
- 🛡️ **Hate Speech Detector** — Flask web app with trained NLP model for detecting toxic content
- 🏠 **Real Estate Price Prediction** — End-to-end pipeline with Flask API and web frontend
- 🎬 **Movie Sentiment Analysis** — NLP-based sentiment classification on movie reviews
- 🌳 **Classical Algorithms** — Decision Tree (Classification & Regression), Random Forest implementations
- 🎨 **Creative ML** — Cartoon-style image transformation using OpenCV
- 🤖 **PAL (Program-Aided Language)** — Experimental notebook on LLM reasoning

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **ML Framework** | Scikit-learn |
| **NLP** | NLTK, TF-IDF |
| **Computer Vision** | OpenCV |
| **Web Framework** | Flask |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter Notebook |

---

## 🏗️ Architecture / How It Works

The repository contains both **standalone notebooks** for algorithm exploration and **full-stack applications** with web interfaces:

```
┌─────────────────────────────────────────────────┐
│           Standalone Notebooks                  │
│  DecisionTree ─ RandomForest ─ Sentiment ─ PAL  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│        Production Applications                  │
│                                                 │
│  Hate Speech Detector    Real Estate Predictor  │
│  ┌─────────────────┐    ┌──────────────────┐   │
│  │ Flask API       │    │ Flask API        │   │
│  │ Trained Model   │    │ Pickle Model     │   │
│  │ Web UI          │    │ HTML/CSS/JS UI   │   │
│  └─────────────────┘    └──────────────────┘   │
│                                                 │
│  Healthcare Prediction                          │
│  ┌──────────────────────┐                      │
│  │ Heart Disease Model  │                      │
│  │ Disease Prediction   │                      │
│  │ Clinical Datasets    │                      │
│  └──────────────────────┘                      │
└─────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Machine-Learning-/
├── ML_Healthcare/                    # Healthcare prediction suite
│   ├── HeartDiseasePrediction.ipynb  # Heart disease classification
│   ├── diseasePrediction.ipynb      # General disease prediction
│   ├── Training.csv                 # Training dataset
│   └── Testing.csv                  # Test dataset
├── hate_speech_detector/            # Flask web app for hate speech
│   ├── app.py                       # Flask server
│   ├── export_model.py              # Model training & export
│   ├── requirements.txt             # Dependencies
│   ├── templates/                   # HTML templates
│   └── static/                      # CSS/JS assets
├── real-estate-price-prediction/    # House price prediction
│   ├── price-prediction.ipynb       # Data science pipeline
│   ├── server.py                    # Flask API
│   ├── util.py                      # Prediction utilities
│   ├── app.html / app.css / app.js  # Web frontend
│   └── *.pickle / *.json            # Trained model artifacts
├── DecisionTreeClassification.ipynb # Decision Tree classifier
├── DecisionTreeRegression.ipynb     # Decision Tree regressor
├── RandomForest.ipynb               # Random Forest ensemble
├── Movie Sentiment Analysis.ipynb   # NLP sentiment classification
├── HateDetection.ipynb              # Hate speech model training
├── Cartoon_opencv.ipynb             # Image cartoonification
├── PAL.ipynb                        # Program-Aided Language
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Vaishnavi-Dubey/Machine-Learning-.git
cd Machine-Learning-

# Install core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter opencv-python flask nltk

# Launch Jupyter for notebooks
jupyter notebook

# Run hate speech detector app
cd hate_speech_detector
pip install -r requirements.txt
python app.py
```

---

## ▶️ Usage

### Notebooks
Open any `.ipynb` file in Jupyter Notebook or Google Colab and run cells sequentially.

### Hate Speech Detector Web App
```bash
cd hate_speech_detector
python app.py
# Visit http://localhost:5000
```

### Real Estate Price Predictor
```bash
cd real-estate-price-prediction
python server.py
# Open app.html in browser
```

---

## 📈 Impact / Learning / Highlights

- 🏥 **Healthcare Impact** — Heart disease prediction achieving meaningful clinical accuracy
- 🛡️ **NLP in Production** — Full-stack hate speech detection with Flask web deployment
- 📊 **End-to-End Pipelines** — From raw data → EDA → feature engineering → model → deployment
- 🧪 **Algorithm Comparison** — Side-by-side evaluation of Decision Trees, Random Forests, and ensemble methods
- 🎯 **Real-World Datasets** — All projects use genuine datasets, not synthetic examples

---

## 🤝 Contributing

Contributions are welcome! Add new algorithms, improve existing models, or enhance documentation.

1. Fork → Branch → Commit → PR

---

## 📜 License

This project is licensed under the **MIT License**.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/Vaishnavi-Dubey">Vaishnavi Dubey</a></b>
</p>
