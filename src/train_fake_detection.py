import pandas as pd
import numpy as np
import os
import re
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")
df["review"] = df["review"].astype(str)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["review_length"] = df["review"].apply(lambda x: len(x.split()))
df["char_length"] = df["review"].apply(len)
df["unique_words"] = df["review"].apply(lambda x: len(set(x.split())))

df["avg_word_length"] = df["review"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)

# 🔥 IMPORTANT FEATURES (balanced)
df["repetition_ratio"] = df["review_length"] / (df["unique_words"] + 1)

df["uppercase_ratio"] = df["review"].apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)

df["punctuation_count"] = df["review"].apply(
    lambda x: len(re.findall(r'[!?.]', x))
)

df["digit_ratio"] = df["review"].apply(
    lambda x: sum(c.isdigit() for c in x) / (len(x) + 1)
)

# -----------------------------
# FEATURES
# -----------------------------
features = [
    'review_length',
    'char_length',
    'unique_words',
    'avg_word_length',
    'repetition_ratio',
    'uppercase_ratio',
    'punctuation_count',
    'digit_ratio'
]

X = df[features].fillna(0)

# -----------------------------
# SCALE
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# MODEL (FINAL BALANCED)
# -----------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.10,   # ✅ tuned
    random_state=42
)

model.fit(X_scaled)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/fake_model.pkl")
joblib.dump(scaler, "models/fake_scaler.pkl")

# -----------------------------
# SCORING
# -----------------------------
scores = model.decision_function(X_scaled)

# -----------------------------
# CLASSIFICATION (TUNED)
# -----------------------------
df["fake_prediction"] = "Genuine"

# Fake (more sensitive)
df.loc[scores < -0.08, "fake_prediction"] = "Fake"

# Ambiguous (narrow range)
df.loc[(scores >= -0.10) & (scores < 0.03), "fake_prediction"] = "Ambiguous"

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("data/processed/fake_detection_results.csv", index=False)

# -----------------------------
# OUTPUT
# -----------------------------
print("\nFake Detection Distribution:\n")
print(df["fake_prediction"].value_counts())

print("\nModel training complete ✅")