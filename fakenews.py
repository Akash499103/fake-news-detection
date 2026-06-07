import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load Dataset ──────────────────────────────────────────
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake['label'] = 0  # 0 = Fake
real['label'] = 1  # 1 = Real

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# ── 2. Preprocessing ─────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)         # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)          # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # remove special characters
    text = re.sub(r'\n', ' ', text)             # remove newlines
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

df['text'] = df['title'] + ' ' + df['text']
df['text'] = df['text'].apply(clean_text)

# ── 3. TF-IDF Vectorization ──────────────────────────────────
x = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# ── 4. Train Multiple Models ─────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

print("Model Performance:\n")
for name, model in models.items():
    model.fit(x_train_tfidf, y_train)
    pred = model.predict(x_test_tfidf)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: {acc*100:.2f}% accuracy")

print("\nDetailed Report (Logistic Regression):")
lr_pred = models["Logistic Regression"].predict(x_test_tfidf)
print(classification_report(y_test, lr_pred, target_names=['Fake', 'Real']))

# ── 5. Test with Custom Input ────────────────────────────────
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    result = models["Logistic Regression"].predict(vectorized)[0]
    return "Real News" if result == 1 else "Fake News"

# Test it
sample = "Scientists discover new planet with signs of life outside solar system"
print(f"\nSample prediction: '{sample}'")
print(f"Result: {predict_news(sample)}")