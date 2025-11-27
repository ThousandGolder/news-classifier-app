import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class NewsClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.categories = self.model.classes_
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict(self, article_text):
        """Predict category for a single article"""
        cleaned_text = self.preprocess_text(article_text)
        prediction = self.model.predict([cleaned_text])[0]
        probabilities = self.model.predict_proba([cleaned_text])[0]
        
        return {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'all_probabilities': dict(zip(self.categories, probabilities.tolist())),
            'top_categories': sorted(
                zip(self.categories, probabilities.tolist()), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        }
    
    def batch_predict(self, articles_list):
        """Predict categories for multiple articles"""
        return [self.predict(article) for article in articles_list]