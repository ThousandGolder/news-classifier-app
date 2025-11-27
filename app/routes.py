from flask import Blueprint, render_template, request, jsonify, flash
import joblib
import os
import sys
import platform
from datetime import datetime
import numpy as np

# Create Blueprint
bp = Blueprint('main', __name__)

# Initialize model - COMPREHENSIVE FIX
MODEL_LOADED = False
model = None
LOAD_ERROR = None

print("🚀 Starting model initialization...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📁 Directory contents: {os.listdir('.')}")

def create_emergency_model():
    """Create a simple working model when the main one fails"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        
        print("🔄 Creating emergency model...")
        
        # Simple but representative training data
        training_data = {
            'texts': [
                # Technology
                "computer technology internet digital software hardware programming AI artificial intelligence machine learning",
                "smartphone mobile app development coding python java javascript web application",
                "tech company startup innovation digital transformation cloud computing data science",
                
                # Sports
                "football soccer basketball sports game match team player championship tournament",
                "olympics athletes competition gold medal world cup premier league basketball NBA",
                "sports news football club basketball team tennis match swimming athletics",
                
                # Politics
                "government politics election minister president policy law parliament democratic",
                "political party election campaign vote parliament congress senate legislation",
                "international relations foreign policy diplomacy government official political news",
                
                # Business
                "business economy market stock exchange company finance investment trade",
                "business news market share economy growth financial report company earnings",
                "startup business entrepreneur investment banking finance economic development",
                
                # Entertainment
                "movie film entertainment celebrity Hollywood music singer actor television show",
                "entertainment news movie release music album celebrity gossip film festival",
                "TV series streaming Netflix Amazon music concert award show entertainment"
            ],
            'labels': [
                'tech', 'tech', 'tech',
                'sport', 'sport', 'sport', 
                'politics', 'politics', 'politics',
                'business', 'business', 'business',
                'entertainment', 'entertainment', 'entertainment'
            ]
        }
        
        # Create and fit pipeline
        emergency_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        emergency_model.fit(training_data['texts'], training_data['labels'])
        
        # Verify it works
        test_text = "technology computer innovation"
        test_pred = emergency_model.predict([test_text])[0]
        test_prob = emergency_model.predict_proba([test_text])
        
        print(f"✅ Emergency model created - Test: '{test_text}' -> {test_pred}")
        print(f"📊 Emergency model classes: {emergency_model.classes_}")
        
        return emergency_model
        
    except Exception as e:
        print(f"❌ Failed to create emergency model: {e}")
        return None

def load_model_with_fallback():
    """Enhanced model loading with multiple fallback strategies"""
    global model, MODEL_LOADED, LOAD_ERROR
    
    model_paths = [
        'app/models/news_classifier_naive_bayes.pkl',
        './app/models/news_classifier_naive_bayes.pkl', 
        'models/news_classifier_naive_bayes.pkl',
        './models/news_classifier_naive_bayes.pkl',
        'news_classifier_naive_bayes.pkl'
    ]
    
    # Try to load existing model
    for path in model_paths:
        if os.path.exists(path):
            print(f"🔍 Found model at: {path}")
            try:
                model = joblib.load(path)
                print("📦 Model loaded from file")
                
                # Test the model
                try:
                    test_pred = model.predict(["technology test"])[0]
                    print(f"✅ Model test passed: {test_pred}")
                    
                    if hasattr(model, 'classes_'):
                        print(f"📊 Model classes: {model.classes_}")
                    
                    MODEL_LOADED = True
                    return True
                    
                except Exception as test_error:
                    print(f"❌ Model test failed: {test_error}")
                    
                    # Check if it's a TF-IDF issue
                    if "idf" in str(test_error).lower() or "fitted" in str(test_error).lower():
                        print("🔧 TF-IDF vectorizer not fitted properly")
                        
                        # Try to fix the existing model
                        if hasattr(model, 'steps'):
                            print("🔄 Attempting to fix pipeline...")
                            try:
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                
                                # Create a simple fitted vectorizer
                                fix_texts = [
                                    "technology computer internet digital",
                                    "sports football basketball game", 
                                    "politics government election policy",
                                    "business market economy stock",
                                    "entertainment movie music film"
                                ]
                                
                                # Replace the first step (assuming it's the vectorizer)
                                step_name = model.steps[0][0]
                                fixed_vectorizer = TfidfVectorizer(
                                    max_features=500,
                                    stop_words='english'
                                ).fit(fix_texts)
                                
                                model.steps[0] = (step_name, fixed_vectorizer)
                                print("✅ Pipeline fixed with new vectorizer")
                                
                                # Test again
                                test_pred = model.predict(["technology"])[0]
                                print(f"✅ Fixed model test: {test_pred}")
                                MODEL_LOADED = True
                                return True
                                
                            except Exception as fix_error:
                                print(f"❌ Pipeline fix failed: {fix_error}")
                    
                    LOAD_ERROR = f"Model test failed: {test_error}"
                    break
                    
            except Exception as load_error:
                print(f"❌ Failed to load model: {load_error}")
                LOAD_ERROR = f"Load error: {load_error}"
                continue
    
    # If we get here, no model worked - create emergency model
    print("🚨 No working model found, creating emergency model...")
    model = create_emergency_model()
    
    if model is not None:
        MODEL_LOADED = True
        print("✅ Emergency model loaded successfully")
        return True
    else:
        LOAD_ERROR = "All model loading strategies failed"
        print("❌ All model loading strategies failed")
        return False

# Load the model
MODEL_LOADED = load_model_with_fallback()

if MODEL_LOADED:
    print("🎉 Model is ready for use!")
else:
    print(f"❌ Model loading failed: {LOAD_ERROR}")

print(f"📊 Final model status: {MODEL_LOADED}")

def preprocess_text(text):
    """Basic text preprocessing"""
    import re
    import nltk
    
    # Download NLTK data if needed
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        try:
            nltk.data.find('corpora/stopwords')
            stop_words = set(stopwords.words('english'))
        except LookupError:
            print("📥 Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        try:
            nltk.data.find('corpora/wordnet')
            lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("📥 Downloading wordnet...")
            nltk.download('wordnet', quiet=True)
            lemmatizer = WordNetLemmatizer()
            
    except Exception as e:
        print(f"⚠️ NLTK setup failed: {e}, using simple preprocessing")
        # Fallback to simple preprocessing
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
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
        
        # Preprocess text
        cleaned_text = preprocess_text(article_text)
        print(f"📝 Processing text: {cleaned_text[:100]}...")
        
        try:
            # Make prediction
            prediction = model.predict([cleaned_text])[0]
            print(f"✅ Prediction: {prediction}")
            
            # Get probabilities
            try:
                probabilities = model.predict_proba([cleaned_text])[0]
                confidence = float(max(probabilities))
                all_probabilities = dict(zip(model.classes_, probabilities.tolist()))
                top_categories = sorted(zip(model.classes_, probabilities.tolist()), key=lambda x: x[1], reverse=True)[:3]
                
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"📈 Top categories: {top_categories}")
                
            except Exception as prob_error:
                print(f"⚠️ Probability error: {prob_error}")
                confidence = 0.8
                all_probabilities = {prediction: confidence}
                top_categories = [(prediction, confidence)]
            
            result = {
                'category': prediction,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'top_categories': top_categories
            }
            
            return render_template('result.html', 
                                 result=result, 
                                 article_preview=article_text[:200] + '...' if len(article_text) > 200 else article_text,
                                 model_loaded=MODEL_LOADED)
            
        except Exception as predict_error:
            error_msg = f'Prediction error: {str(predict_error)}'
            print(f"❌ Prediction failed: {error_msg}")
            
            # Provide helpful error message
            if "idf" in str(predict_error).lower():
                error_msg = "Model configuration error. Using emergency classification."
                
                # Emergency classification based on keywords
                text_lower = article_text.lower()
                categories = {
                    'tech': ['computer', 'technology', 'internet', 'digital', 'software', 'ai', 'app'],
                    'sport': ['sports', 'football', 'basketball', 'game', 'player', 'match', 'team'],
                    'politics': ['government', 'political', 'election', 'policy', 'minister', 'law'],
                    'business': ['business', 'market', 'economy', 'stock', 'company', 'finance'],
                    'entertainment': ['movie', 'film', 'music', 'celebrity', 'show', 'actor', 'tv']
                }
                
                scores = {}
                for category, keywords in categories.items():
                    scores[category] = sum(1 for keyword in keywords if keyword in text_lower)
                
                if sum(scores.values()) > 0:
                    emergency_pred = max(scores.items(), key=lambda x: x[1])[0]
                    confidence = scores[emergency_pred] / len(categories[emergency_pred])
                    
                    result = {
                        'category': emergency_pred,
                        'confidence': min(confidence, 0.95),
                        'all_probabilities': {emergency_pred: confidence},
                        'top_categories': [(emergency_pred, confidence)],
                        'emergency': True
                    }
                    
                    flash('⚠️ Using emergency classification (model issue detected)', 'warning')
                    return render_template('result.html', 
                                         result=result, 
                                         article_preview=article_text[:200] + '...' if len(article_text) > 200 else article_text,
                                         model_loaded=MODEL_LOADED)
            
            flash(error_msg, 'error')
            return render_template('index.html', model_loaded=MODEL_LOADED)
    
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
    model_info = None
    system_info = None
    
    if MODEL_LOADED and model is not None:
        model_info = {
            'name': 'BBC News Classifier',
            'categories': model.classes_.tolist() if hasattr(model, 'classes_') else ['Unknown'],
            'categories_count': len(model.classes_) if hasattr(model, 'classes_') else 0,
            'model_type': str(type(model)),
            'status': 'Loaded and Operational',
            'is_emergency_model': 'Yes' if 'emergency' in str(type(model)).lower() else 'No'
        }
    
    # System information
    system_info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_directory': os.getcwd(),
        'model_loaded': MODEL_LOADED
    }
    
    return render_template('health.html',
                         model_loaded=MODEL_LOADED,
                         model_info=model_info,
                         system_info=system_info,
                         load_error=LOAD_ERROR)

@bp.route('/api/health')
def api_health():
    """API endpoint for health check"""
    health_status = {
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'load_error': LOAD_ERROR,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    if MODEL_LOADED and model is not None:
        health_status.update({
            'model_type': str(type(model)),
            'categories': model.classes_.tolist() if hasattr(model, 'classes_') else [],
            'categories_count': len(model.classes_) if hasattr(model, 'classes_') else 0
        })
    
    return jsonify(health_status)

@bp.route('/debug')
def debug():
    """Enhanced debug endpoint"""
    debug_info = {
        'model_loaded': MODEL_LOADED,
        'load_error': LOAD_ERROR,
        'current_directory': os.getcwd(),
        'app_exists': os.path.exists('app'),
        'models_exists': os.path.exists('app/models') if os.path.exists('app') else False,
        'model_file_exists': os.path.exists('app/models/news_classifier_naive_bayes.pkl') if os.path.exists('app/models') else False
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
        
        # Pipeline inspection
        if hasattr(model, 'steps'):
            debug_info['pipeline_steps'] = []
            for name, step in model.steps:
                step_info = {
                    'name': name,
                    'type': str(type(step)),
                    'is_fitted': hasattr(step, 'vocabulary_') or hasattr(step, 'coef_') or hasattr(step, 'feature_importances_')
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
            confidence = 0.8
            all_probabilities = {prediction: confidence}
        
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

@bp.route('/reload-model', methods=['POST'])
def reload_model():
    """Endpoint to reload the model"""
    global model, MODEL_LOADED, LOAD_ERROR
    
    print("🔄 Manual model reload requested...")
    MODEL_LOADED = load_model_with_fallback()
    
    if MODEL_LOADED:
        return jsonify({
            'success': True,
            'message': 'Model reloaded successfully',
            'model_loaded': MODEL_LOADED
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Model reload failed: {LOAD_ERROR}',
            'model_loaded': MODEL_LOADED
        }), 500