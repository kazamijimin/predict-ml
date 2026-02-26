"""
Emotion Prediction Model
This module trains a machine learning model to predict emotions from text.
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib


class EmotionModel:
    """A machine learning model for predicting emotions from text."""
    
    def __init__(self):
        self.model = None
        self.emotions = ['Stress', 'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Love', 'Neutral']
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_model', 'emotion_model.joblib')
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def train(self, data_path=None):
        """Train the emotion prediction model."""
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'emotion_dataset.csv')
        
        print("Loading dataset...")
        df = pd.read_csv(data_path)  # Comma-separated file
        
        # Preprocess text
        print("Preprocessing text...")
        df['Text'] = df['Text'].apply(self.preprocess_text)
        
        # Split data
        X = df['Text']
        y = df['Emotion']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        # Create pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"\nModel saved to: {self.model_path}")
    
    def load_model(self):
        """Load a pre-trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully!")
            return True
        else:
            print("No saved model found. Please train the model first.")
            return False
    
    def predict(self, text):
        """Predict the emotion for given text."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Get prediction and probabilities
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]
        
        # Get emotion labels and their probabilities
        emotion_probs = dict(zip(self.model.classes_, probabilities))
        
        return prediction, emotion_probs
    
    def get_emotion_info(self, emotion):
        """Get additional information about the predicted emotion."""
        emotion_info = {
            'Stress': {
                'description': 'You may be experiencing stress or pressure.',
                'suggestion': 'Try taking deep breaths, go for a walk, or talk to someone.',
                'color': '\033[91m'  # Red
            },
            'Joy': {
                'description': 'You seem to be feeling happy and joyful!',
                'suggestion': 'Keep spreading the positive vibes!',
                'color': '\033[92m'  # Green
            },
            'Sadness': {
                'description': 'You may be feeling sad or down.',
                'suggestion': 'Its okay to feel this way. Reach out to loved ones or do something you enjoy.',
                'color': '\033[94m'  # Blue
            },
            'Anger': {
                'description': 'You seem to be feeling angry or frustrated.',
                'suggestion': 'Take a moment to cool down. Try counting to 10 or stepping away.',
                'color': '\033[93m'  # Yellow
            },
            'Fear': {
                'description': 'You may be experiencing fear or worry.',
                'suggestion': 'Remember that its normal to feel afraid. Focus on what you can control.',
                'color': '\033[95m'  # Magenta
            },
            'Surprise': {
                'description': 'You seem surprised or caught off guard!',
                'suggestion': 'Embrace the unexpected - it often leads to new opportunities!',
                'color': '\033[96m'  # Cyan
            },
            'Love': {
                'description': 'You seem to be feeling love and affection.',
                'suggestion': 'Cherish these feelings and share them with those you care about!',
                'color': '\033[95m'  # Magenta
            },
            'Neutral': {
                'description': 'Your message seems neutral or factual.',
                'suggestion': 'No strong emotions detected - just everyday communication.',
                'color': '\033[0m'  # Default
            }
        }
        return emotion_info.get(emotion, {
            'description': f'Detected emotion: {emotion}',
            'suggestion': 'Take a moment to reflect on how you are feeling.',
            'color': '\033[0m'
        })


if __name__ == '__main__':
    # Train the model when this script is run directly
    model = EmotionModel()
    model.train()
