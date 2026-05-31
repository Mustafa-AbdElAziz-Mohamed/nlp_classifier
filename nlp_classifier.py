"""
NLP Text Classifier — Customer Support Ticket Classifier
=========================================================
Author  : Mustafa Abdelaziz
Goal    : Automatically classify customer support tickets into categories
Pipeline: Text Cleaning → TF-IDF Vectorization → ML Classifier → REST API
"""

# ─────────────────────────────────────────
# STEP 1: Imports
# ─────────────────────────────────────────
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  NLP TEXT CLASSIFIER — Customer Support Tickets")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 2: Create Dataset
# ─────────────────────────────────────────
tickets = {
    'billing': [
        "I was charged twice for my subscription this month",
        "My invoice shows incorrect amount please fix it",
        "I want a refund for last month's payment",
        "Credit card was charged without my authorization",
        "My payment failed but money was deducted from account",
        "I need a receipt for my recent purchase",
        "Billing statement shows wrong amount please correct",
        "I was overcharged for the premium plan",
        "When will my refund be processed to my bank",
        "How do I update my payment method on file",
        "My credit card expired and I need to change billing",
        "I was billed for a service I cancelled last week",
        "Please send me a copy of all my invoices this year",
        "My account shows a balance I don't recognize",
        "I need to cancel my subscription and get refund",
        "Double payment was made please reverse one transaction",
        "I want to upgrade my plan but worried about billing",
        "Can I get a discount on my monthly invoice",
        "My bank blocked the payment but I want to pay",
        "Please explain the charges on my latest bill",
    ],
    'technical': [
        "The application crashes every time I try to login",
        "I cannot upload files larger than 10MB to the platform",
        "Error 500 appears when I click on the dashboard",
        "The mobile app keeps freezing on Android devices",
        "My data is not syncing between desktop and mobile",
        "API returns 404 error when I call the endpoint",
        "The integration with Slack stopped working today",
        "Password reset link expires before I can use it",
        "Two-factor authentication is not sending SMS code",
        "The export feature downloads an empty CSV file",
        "Charts and graphs are not loading on the reports page",
        "Video call feature has terrible audio quality issues",
        "I cannot connect to the server from my office network",
        "The search function returns no results for any query",
        "My profile picture does not upload successfully ever",
        "Browser extension conflicts with the web application",
        "Notification emails stopped arriving in my inbox",
        "The system is extremely slow during peak hours today",
        "Data backup failed with an unexpected internal error",
        "My API key is not working despite being correct format",
    ],
    'account': [
        "I forgot my password and cannot access my account",
        "Please help me change my username to a new one",
        "I want to delete my account and all my personal data",
        "My account has been locked after too many failed logins",
        "I need to transfer my account to a different email",
        "Can I merge two separate accounts into one single one",
        "I want to update my personal information and address",
        "My account was hacked and I need to recover access",
        "I cannot verify my email because link is expired now",
        "How do I enable two-factor authentication on account",
        "I need to add another admin user to our organization",
        "My account was suspended without any warning or reason",
        "I want to change the email address linked to my account",
        "Please help me recover my account I lost access",
        "Can I create a sub-account for my team members",
        "I accidentally deleted important data from my account",
        "How do I export all my data before closing account",
        "My account shows wrong plan I should be on premium",
        "I need to reset my security questions for my account",
        "Please help me log in from a new device safely",
    ],
    'general': [
        "What are your business hours and support availability",
        "How do I get started with your platform for my team",
        "Can you explain the difference between your pricing plans",
        "What features are included in the enterprise package",
        "How many users can I add to a single account",
        "Do you offer a free trial before purchasing the plan",
        "Is your platform available in Arabic and other languages",
        "What is your data privacy and security policy",
        "Do you provide training or onboarding for new users",
        "How do I contact a human support agent directly",
        "Can I use your service in the United Arab Emirates",
        "What integrations do you support with other tools",
        "How often do you release product updates and features",
        "Is there a mobile app available for iOS and Android",
        "Do you have an affiliate or referral program available",
        "What happens to my data if I cancel my subscription",
        "Can you provide a demo of your platform features",
        "How long does it take to set up the platform initially",
        "Do you offer custom enterprise pricing for large teams",
        "Where can I find your documentation and tutorials",
    ]
}

rows = []
for label, texts in tickets.items():
    for text in texts:
        rows.append({'text': text, 'category': label})

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('support_tickets.csv', index=False)

print(f"\n  STEP 2 — Dataset Created")
print(f"  Total tickets   : {len(df)}")
print(f"  Categories      : {df['category'].unique().tolist()}")
print(f"  Per category    : {dict(df['category'].value_counts())}")

# ─────────────────────────────────────────
# STEP 3: Text Preprocessing
# ─────────────────────────────────────────
def preprocess_text(text):
    """
    Clean and normalize text:
    1. Lowercase everything
    2. Remove punctuation and special characters
    3. Remove extra whitespace
    4. Remove very short words (length < 2)
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if len(w) > 2]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

print(f"\n  STEP 3 — Text Preprocessing")
print(f"  Original : '{df['text'].iloc[0]}'")
print(f"  Cleaned  : '{df['clean_text'].iloc[0]}'")

# ─────────────────────────────────────────
# STEP 4: TF-IDF Vectorization
# ─────────────────────────────────────────
print(f"\n  STEP 4 — TF-IDF Vectorization")

X = df['clean_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),    # unigrams + bigrams
    min_df=1,
    sublinear_tf=True      # apply log scaling to term frequency
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  TF-IDF matrix shape : {X_train_tfidf.shape}")
print(f"  Vocabulary size     : {len(tfidf.vocabulary_)}")

# ─────────────────────────────────────────
# STEP 5: Train Models
# ─────────────────────────────────────────
print(f"\n  STEP 5 — Training Models")

# Model A: Logistic Regression (strong baseline for text)
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_acc  = accuracy_score(y_test, lr_pred)

# Model B: Random Forest on TF-IDF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"  Logistic Regression Accuracy : {lr_acc*100:.1f}%")
print(f"  Random Forest Accuracy       : {rf_acc*100:.1f}%")

# ─────────────────────────────────────────
# STEP 6: Build Full Pipeline (best model)
# ─────────────────────────────────────────
print(f"\n  STEP 6 — Full sklearn Pipeline")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2), sublinear_tf=True)),
    ('clf',   LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)
pipe_pred = pipeline.predict(X_test)
pipe_acc  = accuracy_score(y_test, pipe_pred)

print(f"  Pipeline Accuracy : {pipe_acc*100:.1f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, pipe_pred)}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"  5-Fold CV: {[round(s,3) for s in cv_scores]}  Mean={cv_scores.mean():.3f}")

# ─────────────────────────────────────────
# STEP 7: Visualise
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('NLP Classifier Results', fontsize=13, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, pipe_pred, labels=['billing','technical','account','general'])
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=['billing','technical','account','general'],
            yticklabels=['billing','technical','account','general'])
axes[0].set_title(f'Confusion Matrix (Acc={pipe_acc*100:.0f}%)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].tick_params(axis='x', rotation=30)

# Top features per class
feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
clf = pipeline.named_steps['clf']
classes = clf.classes_
top_n = 6
y_pos = np.arange(top_n)
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

axes[1].set_title('Top TF-IDF Features per Category')
axes[1].axis('off')
table_data = []
for i, cls in enumerate(classes):
    top_idx = clf.coef_[i].argsort()[-top_n:][::-1]
    top_features = [feature_names[j] for j in top_idx]
    table_data.append([cls.upper()] + top_features)

col_labels = ['Category'] + [f'Feature {i+1}' for i in range(top_n)]
tbl = axes[1].table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)

plt.tight_layout()
plt.savefig('nlp_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n  nlp_results.png saved")

# ─────────────────────────────────────────
# STEP 8: Save Model
# ─────────────────────────────────────────
with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("  Model saved → classifier_model.pkl")

# ─────────────────────────────────────────
# STEP 9: Demo Predictions
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO PREDICTIONS")
print("=" * 60)
test_tickets = [
    "I was charged twice this month on my credit card",
    "The app crashes when I try to open the dashboard",
    "I forgot my password and cannot log in",
    "What is the pricing for the enterprise plan",
]
for ticket in test_tickets:
    pred = pipeline.predict([ticket])[0]
    proba = pipeline.predict_proba([ticket]).max()
    print(f"  [{pred.upper():12s}] ({proba*100:.0f}%) → {ticket}")

print("\n  Done! Files: support_tickets.csv | nlp_results.png | classifier_model.pkl")
