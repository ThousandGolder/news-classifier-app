from flask import Blueprint, render_template, request, jsonify, flash
import joblib
import os
import sys
import platform
from datetime import datetime

# Create Blueprint
bp = Blueprint('main', __name__)

# Initialize model - ENHANCED LOADING WITH DEBUGGING
MODEL_LOADED = False
model = None
LOAD_ERROR = None

print("🚀 Starting model initialization...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Directory contents: {os.listdir('.')}")

def load_model_safely():
    """Enhanced model loading with detailed debugging"""
    global model, MODEL_LOADED, LOAD_ERROR
    
    try:
        # Try multiple paths
        model_paths = [
            'app/models/news_classifier_naive_bayes.pkl',
            './app/models/news_classifier_naive_bayes.pkl', 
            'models/news_classifier_naive_bayes.pkl',
            './models/news_classifier_naive_bayes.pkl',
            'news_classifier_naive_bayes.pkl'
        ]
        
        loaded_path = None
        for path in model_paths:
            print(f"🔍 Trying: {path}")
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                loaded_path = path
                try:
                    model = joblib.load(path)
                    print("✅ Model loaded from file!")
                    break
                except Exception as e:
                    print(f"❌ Failed to load from {path}: {e}")
                    continue
        
        if model is None:
            LOAD_ERROR = "Model file not found in any location"
            return False
        
        # Enhanced model validation
        print("🔬 Validating model components...")
        
        # Check if it's a pipeline
        if hasattr(model, 'steps'):
            print("📦 Model is a pipeline")
            for i, (name, step) in enumerate(model.steps):
                print(f"  Step {i}: {name} - {type(step)}")
                
                # Check TF-IDF vectorizer specifically
                if 'vectorizer' in name.lower() or 'tfidf' in name.lower():
                    print(f"  🔍 Checking vectorizer: {name}")
                    if hasattr(step, 'vocabulary_'):
                        print(f"  ✅ Vectorizer has vocabulary_ (fitted)")
                    else:
                        print(f"  ❌ Vectorizer missing vocabulary_ (not fitted)")
                    
                    if hasattr(step, 'idf_'):
                        print(f"  ✅ Vectorizer has idf_ (fitted)")
                    else:
                        print(f"  ❌ Vectorizer missing idf_ (not fitted)")
        
        # Check basic model attributes
        checks = {
            'predict method': hasattr(model, 'predict'),
            'predict_proba method': hasattr(model, 'predict_proba'),
            'classes_ attribute': hasattr(model, 'classes_'),
        }
        
        for check_name, check_result in checks.items():
            status = "✅" if check_result else "❌"
            print(f"{status} {check_name}: {check_result}")
        
        # Test prediction with simple text
        print("🧪 Testing model with sample text...")
        try:
            test_text = "technology computer internet"
            test_prediction = model.predict([test_text])
            print(f"✅ Test prediction successful: {test_prediction[0]}")
            
            if hasattr(model, 'classes_'):
                print(f"📊 Model classes: {model.classes_.tolist()}")
            
            MODEL_LOADED = True
            return True
            
        except Exception as e:
            LOAD_ERROR = f"Model test failed: {str(e)}"
            print(f"❌ Model test failed: {e}")
            
            # Try to diagnose the specific issue
            if "idf" in str(e).lower() or "vector" in str(e).lower():
                print("🔧 Issue detected: TF-IDF vectorizer not properly fitted")
                print("💡 Solution: The model needs to be retrained and saved properly")
            
            return False
            
    except Exception as e:
        LOAD_ERROR = f"Model loading failed: {str(e)}"
        print(f"❌ Model loading failed: {e}")
        return False

# Load the model
MODEL_LOADED = load_model_safely()

if MODEL_LOADED:
    print("🎉 Model is ready for use!")
else:
    print(f"❌ Model loading failed: {LOAD_ERROR}")

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
        flash(f'Model not loaded. Error: {LOAD_ERROR}', 'error')
        return render_template('index.html', model_loaded=MODEL_LOADED)
    
    try:
        article_text = request.form.get('article_text', '').strip()
        
        if not article_text:
            flash('Please enter some text to classify.', 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        if len(article_text) < 10:
            flash('Please enter a longer article text (at least 10 characters).', 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        # Preprocess and predict with enhanced error handling
        cleaned_text = preprocess_text(article_text)
        
        try:
            prediction = model.predict([cleaned_text])[0]
            print(f"✅ Prediction made: {prediction}")
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            print(f"❌ Prediction failed: {error_msg}")
            
            # Provide more specific error message for TF-IDF issues
            if "idf" in str(e).lower():
                error_msg += " - The model's vectorizer is not properly fitted. The model may need to be retrained."
            
            flash(error_msg, 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
        
        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = float(max(probabilities))
            all_probabilities = dict(zip(model.classes_, probabilities.tolist()))
            top_categories = sorted(zip(model.classes_, probabilities.tolist()), key=lambda x: x[1], reverse=True)[:3]
        except Exception as e:
            print(f"⚠️ Probability calculation failed: {e}, using defaults")
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
        
        print(f"🎯 Final prediction: {result['category']} (confidence: {result['confidence']:.2f})")
        
        return render_template('result.html', 
                             result=result, 
                             article_preview=article_text[:200] + '...' if len(article_text) > 200 else article_text,
                             model_loaded=MODEL_LOADED)
    
    except Exception as e:
        error_msg = f'An error occurred during classification: {str(e)}'
        print(f"❌ Classification error: {error_msg}")
        flash(error_msg, 'error')
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
                             system_info=None,
                             load_error=LOAD_ERROR)
    
    # Enhanced model info
    model_info = {
        'name': 'BBC News Classifier',
        'categories': model.classes_.tolist() if hasattr(model, 'classes_') else ['Unknown'],
        'categories_count': len(model.classes_) if hasattr(model, 'classes_') else 0,
        'model_type': str(type(model)),
        'status': 'Loaded and Tested',
        'vectorizer_fitted': 'Yes' if hasattr(model, 'steps') and any(hasattr(step, 'vocabulary_') for _, step in model.steps) else 'Unknown'
    }
    
    # System information
    system_info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'flask_version': '2.3.3',
        'current_directory': os.getcwd()
    }
    
    return render_template('health.html',
                         model_loaded=MODEL_LOADED,
                         model_info=model_info,
                         system_info=system_info,
                         load_error=LOAD_ERROR)

@bp.route('/api/health')
def api_health():
    """API endpoint for health check"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'load_error': LOAD_ERROR,
        'timestamp': datetime.now().isoformat()
    })

@bp.route('/debug')
def debug():
    """Enhanced debug endpoint"""
    debug_info = {
        'model_loaded': MODEL_LOADED,
        'load_error': LOAD_ERROR,
        'current_directory': os.getcwd(),
        'app_exists': os.path.exists('app'),
        'models_exists': os.path.exists('app/models') if os.path.exists('app') else False,
        'model_file_exists': os.path.exists('app/models/news_classifier_naive_bayes.pkl') if os.path.exists('app/models') else False,
        'directory_contents': os.listdir('.')
    }
    
    if MODEL_LOADED and model is not None:
        debug_info.update({
            'model_type': str(type(model)),
            'has_predict': hasattr(model, 'predict'),
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'has_classes': hasattr(model, 'classes_')
        })
        
        if hasattr(model, 'classes_'):
            debug_info['classes'] = model.classes_.tolist()
        
        # Check if it's a pipeline and inspect steps
        if hasattr(model, 'steps'):
            debug_info['pipeline_steps'] = []
            for name, step in model.steps:
                step_info = {
                    'name': name,
                    'type': str(type(step)),
                    'has_vocabulary': hasattr(step, 'vocabulary_'),
                    'has_idf': hasattr(step, 'idf_')
                }
                debug_info['pipeline_steps'].append(step_info)
    
    return jsonify(debug_info)

@bp.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not MODEL_LOADED:
        return jsonify({
            'error': f'Model not loaded: {LOAD_ERROR}',
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text in request body',
                'success': False
            }), 400
        
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({
                'error': 'Text too short (minimum 10 characters)',
                'success': False
            }), 400
        
        cleaned_text = preprocess_text(text)
        prediction = model.predict([cleaned_text])[0]
        
        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = float(max(probabilities))
            all_probabilities = dict(zip(model.classes_, probabilities.tolist()))
        except:
            confidence = 1.0
            all_probabilities = {prediction: 1.0}
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500 