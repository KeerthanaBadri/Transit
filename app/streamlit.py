import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import numpy as np

st.set_page_config(page_title="Transit Insight", layout="wide")

st.title("🚀 Transit Insight Dashboard")

# =============================
# LOAD MODELS (FIXED 🔥)
# =============================
@st.cache_resource
def load_models():
    fake_model = joblib.load("models/fake_model.pkl")
    fake_scaler = joblib.load("models/fake_scaler.pkl")
    return fake_model, fake_scaler

fake_model, fake_scaler = load_models()

# =============================
# LOAD DATA
# =============================
sentiment_df = pd.read_csv("data/processed/sentiment_results.csv")
risk_df = pd.read_csv("data/processed/risk_results.csv")
peak_hours = pd.read_csv("data/processed/peak_hours.csv")
location_risk = pd.read_csv("data/processed/location_risk.csv")
driver_scores = pd.read_csv("data/processed/driver_scores.csv")
fake_df = pd.read_csv("data/processed/fake_detection_results.csv")

trend = pd.read_csv("data/processed/trend.csv")
category = pd.read_csv("data/processed/category.csv")

locations = sorted(fake_df["location"].dropna().unique())
drivers = sorted(fake_df["driver_id"].dropna().unique())

# =============================
# NAVIGATION
# =============================
section = st.sidebar.radio("Navigation", [
    "Overview",
    "Model Comparison",
    "Analytics",
    "Location Analysis",
    "Driver Analysis",
    "Fake Detection",
    "Submit Review"
])

# =============================
# OVERVIEW
# =============================
if section == "Overview":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(fake_df))
    col2.metric("Fake Reviews", fake_df['fake_prediction'].value_counts().get("Fake", 0))
    col3.metric("Drivers", fake_df['driver_id'].nunique())
    col4.metric("Avg Rating", round(fake_df['rating'].mean(), 2))

    st.markdown("### Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(3,3))
    fake_df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# =============================
# MODEL COMPARISON
# =============================
elif section == "Model Comparison":

    combined = pd.concat([
        sentiment_df.assign(Task="Sentiment"),
        risk_df.assign(Task="Risk")
    ])

    st.dataframe(combined)

# =============================
# ANALYTICS
# =============================
elif section == "Analytics":

    st.subheader("Top High Risk Locations")

    fig, ax = plt.subplots()
    sns.barplot(x="high_risk_count", y="location", data=location_risk, ax=ax)
    st.pyplot(fig)

    st.subheader("Peak Hours")

    fig2, ax2 = plt.subplots()
    sns.barplot(x="hour", y="review_count", data=peak_hours, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Trend")

    fig3, ax3 = plt.subplots()
    sns.lineplot(x="time", y="reviews", data=trend, ax=ax3)
    st.pyplot(fig3)

# =============================
# SUBMIT REVIEW (FINAL 🔥)
# =============================
elif section == "Submit Review":

    st.subheader("Submit Review")

    review = st.text_area("Your Review")
    rating = st.slider("Rating", 1, 5)
    location = st.selectbox("Location", locations)
    driver_id = st.selectbox("Driver", drivers)

    if st.button("Submit"):

        if review.strip() == "":
            st.warning("Enter review")
        else:
            st.success("Review submitted!")

            # =============================
            # FEATURE EXTRACTION
            # =============================
            review_len = len(review.split())
            char_len = len(review)
            unique_words = len(set(review.split()))

            avg_word_len = np.mean([len(w) for w in review.split()]) if review.split() else 0

            repetition_ratio = unique_words / (review_len + 1)
            uppercase_ratio = sum(1 for c in review if c.isupper()) / (len(review) + 1)
            punctuation_count = len(re.findall(r'[!?.]', review))
            digit_ratio = sum(c.isdigit() for c in review) / (len(review) + 1)

            features = [[
                review_len,
                char_len,
                unique_words,
                avg_word_len,
                repetition_ratio,
                uppercase_ratio,
                punctuation_count,
                digit_ratio
            ]]

            features_scaled = fake_scaler.transform(features)
            score = fake_model.decision_function(features_scaled)[0]

            # =============================
            # CLASSIFICATION
            # =============================
            if score < -0.2:
                label = "Fake"
            elif score < 0.1:
                label = "Ambiguous"
            else:
                label = "Genuine"

            # =============================
            # OUTPUT
            # =============================
            st.subheader("Prediction")

            col1, col2, col3 = st.columns(3)

            col1.metric("Sentiment", "Not Connected")
            col2.metric("Risk", "Not Connected")
            col3.metric("Authenticity", label)

            # =============================
            # CONFIDENCE
            # =============================
            st.write("Confidence:", round(abs(score), 3))

            # =============================
            # EXPLANATION
            # =============================
            st.markdown("### Why this result?")

            reasons = []

            if repetition_ratio < 0.5:
                reasons.append("Repeated words detected")

            if uppercase_ratio > 0.3:
                reasons.append("Too many uppercase letters")

            if punctuation_count > 3:
                reasons.append("Excess punctuation")

            if digit_ratio > 0.2:
                reasons.append("Contains unusual numbers")

            if not reasons:
                reasons.append("Looks natural")

            for r in reasons:
                st.write("•", r)