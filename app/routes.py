from flask import Blueprint, render_template, request, jsonify, flash
import joblib
import os
import sys
import platform
from datetime import datetime

# Create Blueprint
bp = Blueprint('main', __name__)

# Initialize model - SIMPLIFIED LOADING
MODEL_LOADED = False
model = None

print("🚀 Starting model initialization...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Directory contents: {os.listdir('.')}")

try:
    # Try multiple paths
    model_paths = [
        'app/models/news_classifier_naive_bayes.pkl',
        './app/models/news_classifier_naive_bayes.pkl', 
        'app\\models\\news_classifier_naive_bayes.pkl'
    ]
    
    for path in model_paths:
        print(f"🔍 Trying: {path}")
        if os.path.exists(path):
            print(f"✅ Found model at: {path}")
            try:
                model = joblib.load(path)
                print("✅ Model loaded successfully!")
                
                # Basic checks without strict verification
                if hasattr(model, 'predict'):
                    print("✅ Model has predict method")
                    MODEL_LOADED = True
                    break
                else:
                    print("❌ Model missing predict method")
                    
            except Exception as e:
                print(f"❌ Failed to load from {path}: {e}")
                continue
    
    if MODEL_LOADED:
        print("🎉 Model is ready for use!")
        if hasattr(model, 'classes_'):
            print(f"📊 Model classes: {model.classes_}")
        else:
            print("📊 Model classes: Not available")
            
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    MODEL_LOADED = False

print(f"📊 Final model status: {MODEL_LOADED}")

def preprocess_text(text):
    """Basic text preprocessing"""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    try:
        nltk.data.find('corpora/stopwords')
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    try:
        nltk.data.find('corpora/wordnet')
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('index.html', title='Home', model_loaded=MODEL_LOADED)

@bp.route('/classify', methods=['POST'])
def classify():
    if not MODEL_LOADED:
        flash('Model not loaded. Please check the model file.', 'error')
        return render_template('index.html', model_loaded=MODEL_LOADED)
    
    try:
        article_text = request.form.get('article_text', '').strip()
        
        if not article_text:
            flash('Please enter some text to classify.', 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        if len(article_text) < 10:
            flash('Please enter a longer article text (at least 10 characters).', 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        # Preprocess and predict with error handling
        cleaned_text = preprocess_text(article_text)
        
        try:
            prediction = model.predict([cleaned_text])[0]
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = float(max(probabilities))
            all_probabilities = dict(zip(model.classes_, probabilities.tolist()))
            top_categories = sorted(zip(model.classes_, probabilities.tolist()), key=lambda x: x[1], reverse=True)[:3]
        except:
            # If probabilities fail, use default values
            confidence = 1.0
            all_probabilities = {prediction: 1.0}
            top_categories = [(prediction, 1.0)]
        
        result = {
            'category': prediction,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'top_categories': top_categories
        }
        
        print(f"🎯 Prediction: {result['category']} (confidence: {result['confidence']:.2f})")
        
        return render_template('result.html', 
                             result=result, 
                             article_preview=article_text[:200] + '...' if len(article_text) > 200 else article_text,
                             model_loaded=MODEL_LOADED)
    
    except Exception as e:
        flash(f'An error occurred during classification: {str(e)}', 'error')
        return render_template('index.html', model_loaded=MODEL_LOADED)

@bp.route('/about')
def about():
    return render_template('about.html', title='About', model_loaded=MODEL_LOADED)

@bp.route('/health')
def health():
    """Health check page"""
    if not MODEL_LOADED:
        return render_template('health.html', 
                             model_loaded=MODEL_LOADED,
                             model_info=None,
                             system_info=None)
    
    # Simplified model info
    model_info = {
        'name': 'BBC News Classifier',
        'categories': model.classes_.tolist() if hasattr(model, 'classes_') else ['Unknown'],
        'categories_count': len(model.classes_) if hasattr(model, 'classes_') else 0,
        'model_type': 'Naive Bayes Classifier',
        'vectorizer': 'TF-IDF Vectorizer',
        'accuracy_estimate': 'High accuracy (local testing)'
    }
    
    # System information
    system_info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'flask_version': '2.3.3'
    }
    
    return render_template('health.html',
                         model_loaded=MODEL_LOADED,
                         model_info=model_info,
                         system_info=system_info)

@bp.route('/api/health')
def api_health():
    """API endpoint for health check"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    })

@bp.route('/debug')
def debug():
    """Debug endpoint"""
    debug_info = {
        'model_loaded': MODEL_LOADED,
        'current_directory': os.getcwd(),
        'app_exists': os.path.exists('app'),
        'models_exists': os.path.exists('app/models') if os.path.exists('app') else False,
        'model_file_exists': os.path.exists('app/models/news_classifier_naive_bayes.pkl') if os.path.exists('app/models') else False
    }
    
    if MODEL_LOADED:
        debug_info.update({
            'model_type': str(type(model)),
            'has_predict': hasattr(model, 'predict'),
            'has_classes': hasattr(model, 'classes_')
        })
        
        if hasattr(model, 'classes_'):
            debug_info['classes'] = model.classes_.tolist()
    
    return jsonify(debug_info)