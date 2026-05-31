"""
REST API — NLP Text Classifier
===============================
Exposes the trained classifier as an HTTP endpoint.
Run: python api.py
Test: POST http://localhost:5000/predict  {"text": "I was charged twice"}
"""

from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'classifier_model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'NLP Ticket Classifier API',
        'author': 'Mustafa Abdelaziz',
        'endpoints': {
            'POST /predict': 'Classify a support ticket',
            'GET  /health':  'Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Input  : { "text": "I was charged twice" }
    Output : { "category": "billing", "confidence": 0.94, "all_scores": {...} }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run nlp_classifier.py first.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Send JSON with a "text" field'}), 400

    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text field cannot be empty'}), 400

    prediction   = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes      = model.classes_

    all_scores = {cls: round(float(prob), 4)
                  for cls, prob in zip(classes, probabilities)}

    return jsonify({
        'input_text':  text,
        'category':    prediction,
        'confidence':  round(float(max(probabilities)), 4),
        'all_scores':  all_scores
    })

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Input  : { "texts": ["text1", "text2", ...] }
    Output : list of predictions
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Send JSON with a "texts" list'}), 400

    texts = data['texts']
    predictions   = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for text, pred, proba in zip(texts, predictions, probabilities):
        results.append({
            'text':       text,
            'category':   pred,
            'confidence': round(float(max(proba)), 4)
        })

    return jsonify({'results': results, 'total': len(results)})

if __name__ == '__main__':
    print("\n  NLP Classifier API running at http://localhost:5000")
    print("  Test: curl -X POST http://localhost:5000/predict \\")
    print('         -H "Content-Type: application/json" \\')
    print('         -d \'{"text": "I was charged twice this month"}\'')
    app.run(debug=True, port=5000)
