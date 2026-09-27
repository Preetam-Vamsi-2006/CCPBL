"""
Train sentiment analysis model on IMDB dataset
This script trains a machine learning model for sentiment classification
and saves it for use by the Flask API.

Usage:
    python train_model.py              # Train on full dataset
    python train_model.py --sample     # Train on smaller sample for testing
    python train_model.py --help       # Show help
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import re
import sys
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(sample=False):
    """Load and prepare the dataset"""
    print("=" * 70)
    print("LOADING IMDB DATASET")
    print("=" * 70)
    
    dataset_path = 'IMDB Dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Loading {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    if sample:
        # Use 25% of data for quick testing
        df = df.sample(frac=0.25, random_state=42)
        print(f"Using sample: {len(df)} reviews (25% of dataset)")
    
    print(f"✓ Dataset loaded: {len(df)} reviews")
    print(f"  Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}")
    
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    print("Cleaning reviews...")
    df['review'] = df['review'].apply(clean_text)
    
    # Remove empty reviews after cleaning
    initial_count = len(df)
    df = df[df['review'].str.len() > 0]
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"  Removed {removed_count} empty reviews")
    
    # Convert sentiment to binary (0 for negative, 1 for positive)
    df['sentiment'] = (df['sentiment'] == 'positive').astype(int)
    
    print(f"✓ Preprocessing complete: {len(df)} reviews")
    
    return df

def train_model(X_train, y_train):
    """Train the sentiment analysis model"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    print("Creating TF-IDF vectorizer and Naive Bayes classifier...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', MultinomialNB())
    ])
    
    print(f"Training on {len(X_train)} reviews...")
    pipeline.fit(X_train, y_train)
    print("✓ Model training complete")
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the model on test set"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {tn}")
    print(f"    False Positives: {fp}")
    print(f"    False Negatives: {fn}")
    print(f"    True Positives:  {tp}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(pipeline, metrics):
    """Save the trained model and metrics"""
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    model_path = 'sentiment_model.pkl'
    
    try:
        # Use joblib instead of pickle for better sklearn compatibility
        joblib.dump(pipeline, model_path)
        print(f"✓ Model saved (joblib): {model_path}")
        
        # Save metrics as JSON for easy frontend access
        import json
        from datetime import datetime
        
        metrics_json = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'f1_score': round(metrics['f1'], 4),
            'model_type': 'Multinomial Naive Bayes with TF-IDF',
            'features': {
                'max_features': 5000,
                'ngram_range': [1, 2],
                'min_df': 5,
                'max_df': 0.8
            },
            'dataset': {
                'name': 'IMDB Movie Reviews',
                'total_reviews': 50000,
                'train_split': 0.8,
                'test_split': 0.2,
                'classes': ['negative', 'positive']
            },
            'version': '1.0.0'
        }
        
        # Save as JSON
        metrics_json_path = 'model_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"✓ Metrics saved (JSON): {metrics_json_path}")
        
        # Also save as text for reference
        metrics_txt_path = 'model_metrics.txt'
        with open(metrics_txt_path, 'w') as f:
            f.write("Sentiment Analysis Model Metrics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {metrics_json['timestamp']}\n")
            f.write(f"Model Type: {metrics_json['model_type']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n\n")
            f.write("Features:\n")
            f.write(f"  Max Features: {metrics_json['features']['max_features']}\n")
            f.write(f"  N-gram Range: {metrics_json['features']['ngram_range']}\n")
            f.write(f"  Min DF: {metrics_json['features']['min_df']}\n")
            f.write(f"  Max DF: {metrics_json['features']['max_df']}\n")
        print(f"✓ Metrics saved (TXT): {metrics_txt_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        return False

def test_model(pipeline):
    """Test the model with sample reviews"""
    print("\n" + "=" * 70)
    print("TESTING MODEL WITH SAMPLE REVIEWS")
    print("=" * 70)
    
    samples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible waste of time. The worst film I've ever seen.",
        "It was okay, nothing special but not bad either.",
        "Amazing cinematography and great acting. Highly recommended!",
        "Boring and predictable. I fell asleep halfway through."
    ]
    
    for review in samples:
        cleaned = clean_text(review)
        pred = pipeline.predict([cleaned])[0]
        prob = pipeline.predict_proba([cleaned])[0]
        sentiment = 'POSITIVE' if pred == 1 else 'NEGATIVE'
        confidence = max(prob)
        
        print(f"\n  Review: \"{review[:50]}...\"")
        print(f"  Prediction: {sentiment} (confidence: {confidence:.4f})")

def main():
    """Main training pipeline"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "SENTIMENT ANALYSIS MODEL TRAINING".center(68) + "║")
    print("║" + "IMDB Dataset Classification".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Parse command line arguments
    sample_mode = '--sample' in sys.argv or '-s' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nUsage: python train_model.py [OPTIONS]")
        print("\nOptions:")
        print("  --sample, -s   Train on 25% sample for quick testing")
        print("  --help, -h     Show this help message")
        sys.exit(0)
    
    # Download NLTK data
    download_nltk_data()
    
    # Load dataset
    df = load_dataset(sample=sample_mode)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Split data
    print("\n" + "=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"Training set: {len(X_train)} reviews")
    print(f"Test set:     {len(X_test)} reviews")
    
    # Train model
    pipeline = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Save model
    if save_model(pipeline, metrics):
        # Test with samples
        test_model(pipeline)
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)
        print("\nModel is ready for use!")
        print("Start the Flask app with: python app.py")
    else:
        print("\n" + "=" * 70)
        print("✗ TRAINING FAILED")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    main()
