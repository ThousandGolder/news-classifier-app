# News Article Classification Web App

A Flask web application that classifies news articles into categories using machine learning.

## Features

- Classify news articles into 5 categories: Business, Entertainment, Politics, Sport, Tech
- Real-time classification with confidence scores
- REST API endpoint for programmatic access
- Responsive web interface
- Example articles for testing

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Place your trained model at `models/news_classifier_naive_bayes.pkl`
6. Run the application: `python run.py`

## API Usage

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'