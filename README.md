# 🎬 Sentiment Analysis System

A modern, ML-powered sentiment analysis web application that classifies movie reviews as positive or negative using the IMDB dataset. Built with Flask, scikit-learn, and deployed on Vercel.

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Single Review Analysis**: Analyze individual movie reviews with confidence scores
- **Batch Processing**: Process up to 100 reviews simultaneously
- **Real-time API Status**: Live monitoring of API health and model status
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **REST API**: Full-featured API for integration into other applications
- **Model Information**: Detailed documentation about the ML model
- **Production Ready**: Optimized for Vercel, Heroku, and traditional servers

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the IMDB dataset**
   - Place `IMDB Dataset.csv` in the project root directory
   - Download from: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

5. **Train the model**
   ```bash
   # Full training (takes 5-10 minutes)
   python train_model.py

   # Quick training on 25% sample (takes 1-2 minutes)
   python train_model.py --sample
   ```

6. **Run the development server**
   ```bash
   python app.py
   ```
   
   The application will be available at `http://localhost:5000`

## 📊 Project Structure

```
sentiment-analysis/
├── app.py                      # Flask application
├── wsgi.py                     # WSGI entry point for production
├── train_model.py              # Model training pipeline
├── quick_train.py              # Quick training script
├── explore_dataset.py          # Dataset exploration utility
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for Vercel/Heroku
├── vercel.json                 # Vercel deployment config
├── Procfile                    # Heroku deployment config
├── setup.py                    # Package setup configuration
├── .gitignore                  # Git ignore rules
├── sentiment_model.pkl         # Trained model (generated after training)
├── model_metrics.txt           # Model performance metrics (generated after training)
├── templates/
│   └── index.html             # Web interface
├── static/
│   ├── css/
│   │   └── style.css          # Modern CSS with animations
│   └── js/
│       └── app.js             # Frontend JavaScript
└── README.md                  # This file
```

## 🤖 Model Details

### Architecture
- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 5000
- **N-gram Range**: 1-2 (unigrams and bigrams)
- **Min Document Frequency**: 5
- **Max Document Frequency**: 0.8

### Training Data
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Train/Test Split**: 80/20
- **Classes**: Positive (1) and Negative (0)
- **Average Accuracy**: ~85-88%

### Performance Metrics
After training, you'll see metrics like:
- **Accuracy**: ~0.87 (87%)
- **Precision**: ~0.86
- **Recall**: ~0.89
- **F1-Score**: ~0.87

## 📡 API Documentation

### Base URL
- Local: `http://localhost:5000`
- Production: `https://your-domain.vercel.app`

### Endpoints

#### 1. Analyze Single Review
```http
POST /api/analyze
Content-Type: application/json

{
  "review": "This movie was absolutely amazing! Great acting and plot."
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9487,
  "prediction": 1,
  "text_preview": "this movie was absolutely amazing great acting and plot"
}
```

#### 2. Batch Analysis
```http
POST /api/batch-analyze
Content-Type: application/json

{
  "reviews": [
    "Great movie!",
    "Terrible waste of time",
    "It was okay"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"sentiment": "positive", "confidence": 0.92, "prediction": 1},
    {"sentiment": "negative", "confidence": 0.88, "prediction": 0},
    {"sentiment": "negative", "confidence": 0.65, "prediction": 0}
  ],
  "total": 3,
  "positive_count": 1,
  "negative_count": 2,
  "average_confidence": 0.8167
}
```

#### 3. Get Model Information
```http
GET /api/model-info
```

**Response:**
```json
{
  "model_status": "loaded",
  "model_type": "Naive Bayes with TF-IDF",
  "features": {
    "max_features": 5000,
    "ngram_range": [1, 2],
    "min_df": 5,
    "max_df": 0.8
  },
  "classes": ["negative", "positive"],
  "version": "1.0.0"
}
```

#### 4. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

## 🌐 Deployment

### Deploy to Vercel

1. **Prepare your repository**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [https://vercel.com](https://vercel.com)
   - Sign in with GitHub/GitLab/Bitbucket
   - Click "New Project"
   - Select your repository
   - Click "Deploy"

3. **Configure environment (if needed)**
   - In Vercel dashboard, go to Settings > Environment Variables
   - No special variables required for basic deployment
   - Optional: Set `FLASK_ENV=production`

4. **Train model before deployment**
   - Ensure `sentiment_model.pkl` exists locally
   - Commit and push to trigger deployment

**Important**: The model file must be included in your Git repository or trained during deployment. For production use, consider storing the model in a database or cloud storage if file size is an issue.

### Deploy to Heroku

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Windows (via npm)
   npm install -g heroku
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Train model on Heroku** (optional - if not in repo)
   ```bash
   heroku run python train_model.py
   ```

### Deploy to Traditional VPS

1. **SSH into your server**
   ```bash
   ssh user@your-server.com
   ```

2. **Clone repository**
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

3. **Setup and install**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python train_model.py
   ```

4. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
   ```

5. **Setup Nginx reverse proxy** (optional)
   - Configure Nginx to forward requests to Gunicorn
   - Setup SSL with Let's Encrypt

## 🧪 Testing

### Manual Testing

1. **Test Single Analysis**
   ```bash
   curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"review": "Excellent movie!"}'
   ```

2. **Test Batch Analysis**
   ```bash
   curl -X POST http://localhost:5000/api/batch-analyze \
     -H "Content-Type: application/json" \
     -d '{"reviews": ["Great!", "Bad"]}'
   ```

3. **Test Health Check**
   ```bash
   curl http://localhost:5000/health
   ```

### Web Interface Testing

1. Open `http://localhost:5000` in your browser
2. Test all tabs:
   - **Single Review**: Enter a review and analyze
   - **Batch Analysis**: Enter multiple reviews
   - **API Docs**: Check API documentation

## 🔧 Troubleshooting

### Model not loading
```
Solution: Run python train_model.py to generate sentiment_model.pkl
```

### Import errors
```
Solution: pip install -r requirements.txt
```

### Port already in use
```
Solution: Change port in app.py or kill existing process
python app.py --port 5001
```

### Slow on first request
```
Normal behavior: Model loads on first request (few seconds)
Subsequent requests are fast
```

## 📈 Performance Optimization

### For Production:
1. Use Gunicorn with multiple workers: `gunicorn -w 8 wsgi:app`
2. Add caching headers for static files
3. Use CDN for static assets
4. Consider model quantization for faster inference
5. Implement request rate limiting

### Local Development:
1. Enable debug mode: `FLASK_DEBUG=1 python app.py`
2. Use Flask development server only for testing
3. Install dev dependencies: `pip install flask-debugtoolbar`

## 🔐 Security

### Best Practices Implemented:
- Input validation and sanitization
- CORS enabled for cross-origin requests
- Error handling without exposing sensitive info
- HTML escaping to prevent XSS attacks

### Additional Security:
1. Use HTTPS in production
2. Implement rate limiting
3. Add authentication if needed
4. Use environment variables for secrets
5. Regular dependency updates

## 📝 API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (endpoint doesn't exist) |
| 405 | Method Not Allowed |
| 500 | Server Error |
| 503 | Service Unavailable (model not loaded) |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- IMDB Dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Flask framework for web application
- scikit-learn for machine learning
- NLTK for NLP utilities

## 📞 Support

For issues, questions, or suggestions:
1. Check existing [GitHub Issues](https://github.com/yourrepo/issues)
2. Create a new issue with detailed description
3. Include steps to reproduce if it's a bug
4. Provide system information (OS, Python version, etc.)

## 🔮 Future Enhancements

- [ ] Fine-tuned deep learning models (BERT, RoBERTa)
- [ ] Multi-language support
- [ ] Aspect-based sentiment analysis
- [ ] Emotion detection (happy, sad, angry, etc.)
- [ ] User authentication and dashboard
- [ ] Review history and analytics
- [ ] Model versioning and A/B testing
- [ ] Streaming results for large batches

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies**
