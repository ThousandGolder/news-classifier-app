from flask import Blueprint, render_template, request, jsonify, flash
import joblib
import os
import sys
import platform
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create Blueprint
bp = Blueprint('main', __name__)

# Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models', 'news_classifier_naive_bayes.pkl')

print(f"🔍 Looking for model at: {MODEL_PATH}")

# Initialize classifier with verification
try:
    MODEL_PATH = 'app/models/news_classifier_naive_bayes.pkl'
    
    if os.path.exists(MODEL_PATH):
        print("🔍 Loading model...")
        model = joblib.load(MODEL_PATH)
        
        # Verify the model components are properly loaded
        try:
            # Test if vectorizer is fitted by trying a simple transform
            test_text = "test business"
            _ = model.named_steps['tfidf'].transform([test_text])
            
            # Test if classifier works
            _ = model.predict([test_text])
            
            MODEL_LOADED = True
            print("✅ Model loaded and verified successfully!")
            print(f"📊 Model classes: {model.classes_}")
            
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            MODEL_LOADED = False
            
    else:
        MODEL_LOADED = False
        print(f"❌ Model file not found at: {MODEL_PATH}")
        # List files in models directory
        models_dir = os.path.join(BASE_DIR, 'app', 'models')
        if os.path.exists(models_dir):
            print("📁 Files in models directory:")
            for f in os.listdir(models_dir):
                print(f"   - {f}")
        else:
            print("❌ Models directory doesn't exist")
        
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

def preprocess_text(text):
    """Basic text preprocessing"""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download NLTK data if needed
    try:
        nltk.data.find('corpora/stopwords')
        stop_words = set(stopwords.words('english'))
    except LookupError:
        print("📥 Downloading NLTK stopwords...")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    try:
        nltk.data.find('corpora/wordnet')
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        print("📥 Downloading NLTK wordnet...")
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    
    if not isinstance(text, str):
        return ""
    
    # Clean text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words 
            if word not in stop_words and len(word) > 2]
    
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
        
        # Preprocess and predict
        cleaned_text = preprocess_text(article_text)
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        
        # Create result dictionary
        result = {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'all_probabilities': dict(zip(model.classes_, probabilities.tolist())),
            'top_categories': sorted(
                zip(model.classes_, probabilities.tolist()), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
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
    """Health check page with detailed system information"""
    if not MODEL_LOADED:
        return render_template('health.html', 
                             model_loaded=MODEL_LOADED,
                             model_info=None,
                             system_info=None)
    
    # Model information (without accessing potentially un-fitted vectorizer)
    model_info = {
        'name': 'BBC News Classifier',
        'algorithm': type(model.named_steps['model']).__name__,
        'categories': model.classes_.tolist(),
        'categories_count': len(model.classes_),
        'model_type': 'Naive Bayes Classifier',
        'vectorizer': 'TF-IDF Vectorizer',
        'accuracy_estimate': '99.1% (on test data)'
    }
    
    # Only try to get feature count if vectorizer is fitted
    try:
        model_info['features_count'] = len(model.named_steps['tfidf'].get_feature_names_out())
    except:
        model_info['features_count'] = 'Unknown (vectorizer not fitted)'
    
    # System information
    import sys
    
    system_info = {
        'python_version': sys.version.split()[0],
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
    health_data = {
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    }
    
    if MODEL_LOADED:
        health_data.update({
            'categories': model.classes_.tolist(),
            'categories_count': len(model.classes_),
            'model_algorithm': type(model.named_steps['model']).__name__
        })
    
    return jsonify(health_data)

@bp.route('/model-debug')
def model_debug():
    """Debug route to check model state"""
    debug_info = {
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    }
    
    if MODEL_LOADED:
        debug_info.update({
            'has_tfidf': 'tfidf' in model.named_steps,
            'has_model': 'model' in model.named_steps,
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
        })
        
        # Check vectorizer state
        if 'tfidf' in model.named_steps:
            tfidf = model.named_steps['tfidf']
            debug_info.update({
                'tfidf_has_vocabulary': hasattr(tfidf, 'vocabulary_'),
                'tfidf_has_idf': hasattr(tfidf, 'idf_'),
                'tfidf_has_feature_names': hasattr(tfidf, 'get_feature_names_out')
            })
    
    return jsonify(debug_info)

# Debug route to check model status
@bp.route('/debug')
def debug():
    return jsonify({
        'model_loaded': MODEL_LOADED,
        'model_exists': os.path.exists(MODEL_PATH),
        'current_directory': os.getcwd()
    })