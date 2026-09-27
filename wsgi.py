"""
WSGI entry point for production deployment (Vercel, Heroku, etc.)
This file is used by application servers (Gunicorn, Vercel, etc.) to run the Flask app
"""

import os
import sys
import pickle
import re
import json
from pathlib import Path

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import app

# Global model reference
_model = None
_model_loaded = False

def load_model_on_startup():
    """Load the model when the WSGI app starts"""
    global _model, _model_loaded
    
    model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                _model = pickle.load(f)
            print("✓ Model loaded successfully in WSGI")
            _model_loaded = True
            # Set the model in the app context
            app.model = _model
        except Exception as e:
            print(f"✗ Error loading model in WSGI: {str(e)}")
            _model_loaded = False
    else:
        print(f"⚠ Warning: Model file not found at {model_path}")
        print("  Sentiment analysis will not work until model is trained")
        _model_loaded = False

# Middleware to ensure model is loaded
@app.before_request
def ensure_model():
    """Ensure model is loaded before processing requests"""
    if not _model_loaded and not hasattr(app, 'model'):
        load_model_on_startup()

# Initialize the model on startup
if __name__ != '__main__':
    # This runs when the WSGI server starts (not when file is run directly)
    load_model_on_startup()

# Export the app for WSGI servers
if __name__ == '__main__':
    app.run()

