"""
WSGI entry point for production deployment (Vercel, Heroku, etc.)
This file is used by application servers (Gunicorn, Vercel, etc.) to run the Flask app
"""

import os
import sys
import pickle
import json
from pathlib import Path

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import app

# Global model reference - set to app globals
_model_loaded = False

def load_model_on_startup():
    """Load the model when the WSGI app starts"""
    global _model_loaded
    
    # Use absolute path for Vercel compatibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'sentiment_model.pkl')
    metrics_path = os.path.join(current_dir, 'model_metrics.json')
    
    print("=" * 70)
    print("🔍 WSGI STARTUP: Looking for model")
    print("=" * 70)
    print(f"🔍 WSGI Startup: Looking for model at {model_path}")
    
    if os.path.exists(model_path):
        try:
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"📦 Model file found, size: {size_mb:.2f} MB")
            with open(model_path, 'rb') as f:
                _model = pickle.load(f)
            print("✅ Model loaded successfully in WSGI")
            _model_loaded = True
            # Set the model in the app context
            app.model = _model
            app.model_ready = True
            
            # Load metrics too
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        app.metrics = json.load(f)
                    print("✅ Metrics loaded successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load metrics: {str(e)}")
        except Exception as e:
            print(f"❌ Error loading model in WSGI: {str(e)}")
            import traceback
            traceback.print_exc()
            _model_loaded = False
            app.model_ready = False
    else:
        print(f"❌ Model file NOT found at {model_path}")
        print(f"   Directory contents:")
        try:
            contents = os.listdir(current_dir)
            for item in contents[:30]:
                full_path = os.path.join(current_dir, item)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path) / 1024
                    print(f"     - {item} ({size:.1f} KB)")
                else:
                    print(f"     - {item}/ (dir)")
        except Exception as e:
            print(f"   Error: {e}")
        _model_loaded = False
        app.model_ready = False
    
    print("=" * 70 + "\n")

# Middleware to ensure model is loaded
@app.before_request
def ensure_model():
    """Ensure model is loaded before processing requests"""
    if not _model_loaded or not app.model:
        load_model_on_startup()

# Initialize the model on startup
if __name__ != '__main__':
    # This runs when the WSGI server starts (not when file is run directly)
    print("\n🚀 WSGI Server Starting...")
    load_model_on_startup()

# Export the app for WSGI servers
if __name__ == '__main__':
    app.run()

