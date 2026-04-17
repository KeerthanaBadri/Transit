import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/final_dataset.csv")

print("Initial Shape:", df.shape)

# -----------------------------
# 2. Fix Numeric Columns ⚠️
# -----------------------------
numeric_cols = [
    'review_length', 'char_length',
    'unique_words', 'avg_word_length'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# 3. Clean Text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

# -----------------------------
# 4. Encode Labels
# -----------------------------
label_encoders = {}

categorical_cols = ['sentiment', 'risk', 'category', 'authenticity']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# 5. Save Clean Dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Cleaned data saved ✅")

# -----------------------------
# 6. Save Encoders (Optional)
# -----------------------------
import pickle

with open("data/processed/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Encoders saved ✅")