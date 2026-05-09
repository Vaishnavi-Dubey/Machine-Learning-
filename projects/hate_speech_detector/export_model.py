import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and prepare the data
df = pd.read_csv('labeled_data.csv')
df = df[['class', 'tweet']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], 
    df['class'], 
    test_size=0.2, 
    random_state=42
)

# Create and fit the vectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer have been saved successfully!") 