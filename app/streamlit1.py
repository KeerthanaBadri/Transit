import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import numpy as np

st.set_page_config(page_title="Transit Insight", layout="wide")

st.title("🚀 Transit Insight Dashboard")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_fake_model():
    model = joblib.load("models/fake_model.pkl")
    scaler = joblib.load("models/fake_scaler.pkl")
    return model, scaler

fake_model, fake_scaler = load_fake_model()

# -----------------------------
# LOAD DATA
# -----------------------------
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

# -----------------------------
# TEST BUTTONS 🔥
# -----------------------------
st.markdown("### 🧪 Quick Test")

colA, colB, colC = st.columns(3)

if colA.button("Fake Example"):
    st.session_state.review = "good good good good"

if colB.button("Genuine Example"):
    st.session_state.review = "Nice ride and polite driver"

if colC.button("Ambiguous Example"):
    st.session_state.review = "ok"

# -----------------------------
# NAVIGATION
# -----------------------------
section = st.sidebar.radio("Navigation", [
    "Overview",
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

    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(fake_df))
    col2.metric("Fake Reviews", fake_df['fake_prediction'].value_counts().get("Fake", 0))
    col3.metric("Drivers", fake_df['driver_id'].nunique())
    col4.metric("Avg Rating", round(fake_df['rating'].mean(), 2))

    st.markdown("---")

    st.subheader("🙂 Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(3,3))
    fake_df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# =============================
# ANALYTICS
# =============================
elif section == "Analytics":

    st.subheader("📍 High Risk Locations")

    fig, ax = plt.subplots()
    sns.barplot(x="high_risk_count", y="location", data=location_risk, ax=ax)
    st.pyplot(fig)

    st.caption("Higher value → more high-risk incidents")

    st.subheader("⏱️ Peak Hours")

    fig2, ax2 = plt.subplots()
    sns.barplot(x="hour", y="review_count", data=peak_hours, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Trend")

    fig3, ax3 = plt.subplots()
    sns.lineplot(x="time", y="reviews", data=trend, ax=ax3)
    st.pyplot(fig3)

# =============================
# LOCATION ANALYSIS
# =============================
elif section == "Location Analysis":

    loc = st.selectbox("Select Location", locations)
    df_loc = fake_df[fake_df["location"] == loc]

    st.subheader(loc)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df_loc['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        df_loc['risk'].value_counts().plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

# =============================
# DRIVER ANALYSIS
# =============================
elif section == "Driver Analysis":

    st.dataframe(driver_scores.head(10))

# =============================
# FAKE DETECTION
# =============================
elif section == "Fake Detection":

    col1, col2 = st.columns([1,2])

    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        fake_df['fake_prediction'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    with col2:
        st.info("""
        Genuine → Real reviews  
        Fake → Spam / abnormal  
        Ambiguous → borderline  
        """)

# =============================
# SUBMIT REVIEW (FINAL 🔥)
# =============================
elif section == "Submit Review":

    st.subheader("📝 Submit Review")

    review = st.text_area("Your Review", value=st.session_state.get("review", ""))
    rating = st.slider("Rating", 1, 5)
    location = st.selectbox("Location", locations)
    driver = st.selectbox("Driver", drivers)

    if st.button("Analyze"):

        if review.strip() == "":
            st.warning("Enter review")
        else:
            st.success("Processing...")

            # -----------------------------
            # FEATURES
            # -----------------------------
            review_len = len(review.split())
            char_len = len(review)
            unique_words = len(set(review.split()))

            avg_word_len = np.mean([len(w) for w in review.split()]) if review.split() else 0

            repetition_ratio = review_len / (unique_words + 1)
            uppercase_ratio = sum(1 for c in review if c.isupper()) / (len(review) + 1)
            punctuation_count = len(re.findall(r'[!?.]', review))
            digit_ratio = sum(c.isdigit() for c in review) / (len(review) + 1)

            features_df = pd.DataFrame([{
                "review_length": review_len,
                "char_length": char_len,
                "unique_words": unique_words,
                "avg_word_length": avg_word_len,
                "repetition_ratio": repetition_ratio,
                "uppercase_ratio": uppercase_ratio,
                "punctuation_count": punctuation_count,
                "digit_ratio": digit_ratio
            }])

            scaled = fake_scaler.transform(features_df)
            score = fake_model.decision_function(scaled)[0]

            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            if repetition_ratio > 1.8:
                label = "Fake"
            elif score < -0.10:
                label = "Fake"
            elif score < 0.03:
                label = "Ambiguous"
            else:
                label = "Genuine"

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.markdown("---")
            st.subheader("🤖 Prediction")

            if label == "Fake":
                st.error(label)
            elif label == "Ambiguous":
                st.warning(label)
            else:
                st.success(label)

            confidence = round(abs(score), 3)
            st.write("Confidence:", confidence)
            st.progress(min(abs(score), 1.0))

            # -----------------------------
            # EXPLANATION
            # -----------------------------
            st.markdown("### 🔍 Explanation")

            st.write(f"Repetition Ratio: {round(repetition_ratio,2)}")
            st.write(f"Punctuation Count: {punctuation_count}")
            st.write(f"Uppercase Ratio: {round(uppercase_ratio,2)}")

# -----------------------------
# DOWNLOAD
# -----------------------------
st.markdown("---")
st.download_button("Download Dataset", fake_df.to_csv(index=False))