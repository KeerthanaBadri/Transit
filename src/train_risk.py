import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lightgbm import LGBMClassifier

# -----------------------------
# 1. Load Clean Data
# -----------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")

# Fix text issues
df['clean_text'] = df['clean_text'].fillna("")
df = df[df['clean_text'].str.strip() != ""]

X = df['clean_text']
y = df['risk']

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 4. Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf = rf_model.predict(X_test_tfidf)

# -----------------------------
# 5. LightGBM Model
# -----------------------------
lgb_model = LGBMClassifier()
lgb_model.fit(X_train_tfidf, y_train)

y_pred_lgb = lgb_model.predict(X_test_tfidf)

# -----------------------------
# 6. Evaluation Function
# -----------------------------
def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted')
    }

rf_results = evaluate(y_test, y_pred_rf)
lgb_results = evaluate(y_test, y_pred_lgb)

# -----------------------------
# 7. Print Results
# -----------------------------
print("\nRandom Forest Results:")
print(rf_results)

print("\nLightGBM Results:")
print(lgb_results)

# -----------------------------
# 8. Save Results
# -----------------------------
results_df = pd.DataFrame({
    "Model": ["Random Forest", "LightGBM"],
    "Accuracy": [rf_results["Accuracy"], lgb_results["Accuracy"]],
    "Precision": [rf_results["Precision"], lgb_results["Precision"]],
    "Recall": [rf_results["Recall"], lgb_results["Recall"]],
    "F1 Score": [rf_results["F1 Score"], lgb_results["F1 Score"]],
})

results_df.to_csv("data/processed/risk_results.csv", index=False)

print("\nRisk results saved to risk_results.csv ✅")