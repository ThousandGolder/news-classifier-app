import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    MODEL_PATH = 'models/news_classifier_naive_bayes.pkl'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size