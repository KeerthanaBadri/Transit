import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

# -----------------------------
# 1. Load Clean Data
# -----------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")
df['clean_text'] = df['clean_text'].fillna("")
df = df[df['clean_text'].str.strip() != ""]
X = df['clean_text']
y = df['sentiment']

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
# 4. Logistic Regression Model
# -----------------------------
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)

# -----------------------------
# 5. XGBoost Model
# -----------------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_tfidf, y_train)

y_pred_xgb = xgb_model.predict(X_test_tfidf)

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

lr_results = evaluate(y_test, y_pred_lr)
xgb_results = evaluate(y_test, y_pred_xgb)

# -----------------------------
# 7. Show Results
# -----------------------------
print("\nLogistic Regression Results:")
print(lr_results)

print("\nXGBoost Results:")
print(xgb_results)

# -----------------------------
# 8. Save Results
# -----------------------------
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "Accuracy": [lr_results["Accuracy"], xgb_results["Accuracy"]],
    "Precision": [lr_results["Precision"], xgb_results["Precision"]],
    "Recall": [lr_results["Recall"], xgb_results["Recall"]],
    "F1 Score": [lr_results["F1 Score"], xgb_results["F1 Score"]],
})

results_df.to_csv("data/processed/sentiment_results.csv", index=False)

print("\nResults saved to sentiment_results.csv ✅")