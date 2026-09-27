"""
Flask Application for Sentiment Analysis
API endpoints for sentiment analysis using trained ML model
"""

import os
import pickle
import re
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global variable to store the model
model = None
metrics = None

def load_model():
    """Load the pre-trained sentiment analysis model"""
    global model, metrics
    model_path = 'sentiment_model.pkl'
    metrics_path = 'model_metrics.json'
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully")
            
            # Load metrics if available
            if os.path.exists(metrics_path):
                try:
                    import json
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    print("✓ Model metrics loaded successfully")
                except Exception as e:
                    print(f"⚠ Warning: Could not load metrics: {str(e)}")
                    metrics = None
            else:
                print("⚠ Warning: Model metrics file not found")
                metrics = None
            
            return True
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Please run 'python train_model.py' first")
        return False

def clean_text(text):
    """Clean and preprocess text for prediction"""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(review_text):
    """
    Analyze sentiment of the given review text
    Returns: {sentiment, confidence, prediction}
    """
    if model is None:
        return {
            'error': 'Model not loaded',
            'sentiment': None,
            'confidence': 0,
            'message': 'Please train the model first'
        }
    
    try:
        # Clean the text
        cleaned_text = clean_text(review_text)
        
        if not cleaned_text:
            return {
                'error': 'Empty review after cleaning',
                'sentiment': None,
                'confidence': 0
            }
        
        # Get prediction
        prediction = model.predict([cleaned_text])[0]
        
        # Get prediction probability
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))
        
        # Map prediction to label
        sentiment_label = 'positive' if prediction == 1 else 'negative'
        
        return {
            'sentiment': sentiment_label,
            'confidence': round(confidence, 4),
            'prediction': int(prediction),
            'text_preview': cleaned_text[:100] + ('...' if len(cleaned_text) > 100 else '')
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'sentiment': None,
            'confidence': 0
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serve the frontend interface"""
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api_home():
    """API information"""
    return jsonify({
        'app': 'Sentiment Analysis API',
        'version': '1.0.0',
        'description': 'Sentiment analysis using ML model trained on IMDB dataset',
        'endpoints': {
            'GET /': 'Web interface',
            'GET /api': 'API information (this)',
            'GET /health': 'Health check',
            'POST /api/analyze': 'Analyze sentiment of review text',
            'POST /api/batch-analyze': 'Analyze multiple reviews',
            'GET /api/model-info': 'Get model information'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'timestamp': pd.Timestamp.now().isoformat()
    }), 200 if model_loaded else 503

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if metrics:
        return jsonify(metrics), 200
    
    # Fallback if metrics not available
    return jsonify({
        'model_status': 'loaded',
        'model_type': 'Naive Bayes with TF-IDF',
        'features': {
            'max_features': 5000,
            'ngram_range': [1, 2],
            'min_df': 5,
            'max_df': 0.8
        },
        'classes': ['negative', 'positive'],
        'version': '1.0.0'
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze sentiment of a single review
    
    Request JSON:
    {
        "review": "Review text here"
    }
    
    Response JSON:
    {
        "sentiment": "positive|negative",
        "confidence": 0.95,
        "prediction": 1|0
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({
                'error': 'Missing "review" field',
                'message': 'Please provide a "review" field in JSON body'
            }), 400
        
        review_text = data['review']
        
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({
                'error': 'Invalid review',
                'message': 'Review must be a non-empty string'
            }), 400
        
        result = analyze_sentiment(review_text)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze sentiment of multiple reviews
    
    Request JSON:
    {
        "reviews": ["Review 1", "Review 2", ...]
    }
    
    Response JSON:
    {
        "results": [
            {"sentiment": "positive", "confidence": 0.95},
            ...
        ],
        "total": 2,
        "positive_count": 1,
        "negative_count": 1
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({
                'error': 'Missing "reviews" field',
                'message': 'Please provide a "reviews" array in JSON body'
            }), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list) or len(reviews) == 0:
            return jsonify({
                'error': 'Invalid reviews',
                'message': 'Reviews must be a non-empty array'
            }), 400
        
        # Limit batch size
        if len(reviews) > 100:
            return jsonify({
                'error': 'Batch too large',
                'message': 'Maximum 100 reviews per batch'
            }), 400
        
        results = []
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            if not isinstance(review, str) or not review.strip():
                results.append({
                    'sentiment': None,
                    'confidence': 0,
                    'error': 'Invalid review'
                })
            else:
                result = analyze_sentiment(review)
                results.append(result)
                
                if 'sentiment' in result:
                    if result['sentiment'] == 'positive':
                        positive_count += 1
                    else:
                        negative_count += 1
        
        return jsonify({
            'results': results,
            'total': len(results),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'average_confidence': round(
                np.mean([r.get('confidence', 0) for r in results if 'confidence' in r]),
                4
            )
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /api/model-info',
            'POST /api/analyze',
            'POST /api/batch-analyze'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method Not Allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SENTIMENT ANALYSIS API")
    print("=" * 60)
    
    # Import pandas for timestamp
    import pandas as pd
    
    # Load the model
    if load_model():
        print("\nStarting Flask server...")
        print("Server running on http://localhost:5000")
        print("API documentation: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Cannot start server without model")
        print("Please train the model first: python train_model.py")
