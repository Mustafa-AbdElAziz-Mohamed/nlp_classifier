# 🗂️ NLP Text Classifier — Customer Support Tickets

## What This Project Does
Automatically classifies customer support tickets into 4 categories using NLP and machine learning.  
The trained model is served as a **REST API** using Flask.

## Categories
| Category | Example |
|---|---|
| `billing` | "I was charged twice this month" |
| `technical` | "The app crashes on login" |
| `account` | "I forgot my password" |
| `general` | "What are your business hours?" |

## Full Pipeline

```
Raw Text
   ↓
Text Cleaning (lowercase, remove punctuation, remove short words)
   ↓
TF-IDF Vectorization (converts words to numbers)
   ↓
Logistic Regression Classifier
   ↓
Prediction + Confidence Score
   ↓
REST API (Flask)
```

## How to Run

### Step 1: Train the model
```bash
pip install scikit-learn pandas numpy matplotlib seaborn flask
python nlp_classifier.py
```

### Step 2: Start the API
```bash
python api.py
```

### Step 3: Test the API
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month"}'

# Response:
{
  "category": "billing",
  "confidence": 0.94,
  "input_text": "I was charged twice this month",
  "all_scores": { "billing": 0.94, "technical": 0.02, "account": 0.02, "general": 0.02 }
}

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["app crashes", "I forgot my password"]}'
```

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Classify one ticket |
| POST | `/predict/batch` | Classify multiple tickets |

## Key Concepts Used
- **TF-IDF**: converts text to numerical features weighted by importance
- **Bigrams**: captures two-word phrases ("credit card", "password reset")
- **Pipeline**: chains preprocessing + model into one object
- **Pickle**: saves the trained model to disk for reuse
- **Flask REST API**: serves predictions over HTTP

## Results
- Accuracy: ~95%+
- 5-Fold Cross-Validation: ~94% mean

## Author
Mustafa Abdelaziz — AI & Data Engineer
https://github.com/Mustafa-AbdElAziz-Mohamed
