import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import numpy as np
@st.cache_resource
def load_fake_model():
    model = joblib.load("models/fake_model.pkl")
    scaler = joblib.load("models/fake_scaler.pkl")
    return model, scaler

fake_model, fake_scaler = load_fake_model()

st.set_page_config(page_title="Transit Insight", layout="wide")

st.title("🚀 Transit Insight Dashboard")

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
# NAVIGATION
# -----------------------------
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

    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(fake_df))
    fake_counts = fake_df['fake_prediction'].value_counts()

    col2.metric("Fake Reviews", fake_counts.get("Fake", 0))
    col3.metric("Drivers", fake_df['driver_id'].nunique())
    col4.metric("Avg Rating", round(fake_df['rating'].mean(), 2))

    st.markdown("### 🙂 Sentiment Distribution")

    colA, colB = st.columns([1,2])

    with colA:
        fig, ax = plt.subplots(figsize=(3,3))
        fake_df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    with colB:
        st.info("""
        • Positive → Good service  
        • Negative → Complaints  
        • Neutral → Mixed feedback  
        """)

    peak = peak_hours.sort_values("review_count", ascending=False).iloc[0]

    st.markdown("### 📊 Insights")
    st.write(f"• Peak hour: **{peak['hour']}** with {peak['review_count']} reviews")
    st.write(f"• Most risky location: **{location_risk.iloc[0]['location']}**")

    st.markdown("### 💡 Recommendations")
    st.write("""
    1. Focus on high-risk locations  
    2. Retrain high-risk drivers  
    3. Improve service quality  
    4. Monitor fake reviews  
    """)

# =============================
# MODEL COMPARISON
# =============================
elif section == "Model Comparison":

    combined = pd.concat([
        sentiment_df.assign(Task="Sentiment"),
        risk_df.assign(Task="Risk")
    ])

    st.subheader("📊 Model Performance Table")
    st.dataframe(combined)

    st.subheader("📈 Model Comparison")

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for i in range(0, len(metrics), 2):
        col1, col2 = st.columns(2)

        for col, metric in zip([col1, col2], metrics[i:i+2]):
            with col:
                fig, ax = plt.subplots()
                sns.barplot(x="Model", y=metric, hue="Task", data=combined, ax=ax)
                ax.set_title(metric)
                st.pyplot(fig)

# =============================
# ANALYTICS (FINAL CLEAN)
# =============================
elif section == "Analytics":

    st.subheader("📍 Top 10 High Risk Locations")

    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.barplot(x="high_risk_count", y="location", data=location_risk, ax=ax1)
    st.pyplot(fig1)

    st.caption("Higher bar = more high-risk incidents in that location")

    st.subheader("⏱️ Peak Hour Activity")

    fig2, ax2 = plt.subplots()
    sns.barplot(x="hour", y="review_count", data=peak_hours, ax=ax2)
    st.pyplot(fig2)

    st.caption("Hour (0–23) vs number of reviews — shows busiest times")

    st.subheader("📈 Trend Over Time")

    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.lineplot(x="time", y="reviews", data=trend, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.caption("Shows how number of reviews changes over time")

    st.subheader("📊 Complaint Categories")

    col1, col2 = st.columns([2,1])

    with col1:
        fig4, ax4 = plt.subplots(figsize=(6,3))
        sns.barplot(x="count", y="category", data=category, ax=ax4)
        st.pyplot(fig4)

    with col2:
        st.info("""
        • Delay → Late service  
        • Pricing → Fare issues  
        • Driver Behavior → Driver issues  
        • Safety → Risk incidents  
        • Vehicle Condition → Vehicle problems  
        """)

# =============================
# LOCATION ANALYSIS (FINAL FIX)
# =============================
elif section == "Location Analysis":

    selected_location = st.selectbox("Select Location", locations)
    df_loc = fake_df[fake_df["location"] == selected_location]

    st.subheader(f"📍 {selected_location}")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        df_loc['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
        st.caption("Sentiment distribution")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        df_loc['risk'].value_counts().plot(kind='bar', ax=ax2)
        st.pyplot(fig2)
        st.caption("Risk levels (0=Low, 1=Medium, 2=High)")

    # SAME ROW FIX
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### ⏱️ Peak Hours")

        loc_peak = df_loc.groupby("hour").size().reset_index(name="count")

        fig3, ax3 = plt.subplots(figsize=(5,3))
        sns.barplot(x="hour", y="count", data=loc_peak, ax=ax3)
        st.pyplot(fig3)

        st.caption("Hour vs number of reviews in this location")

    with col4:
        st.markdown("### 📊 Categories")

        loc_cat = df_loc["category"].value_counts().reset_index()
        loc_cat.columns = ["category", "count"]

        fig4, ax4 = plt.subplots(figsize=(5,3))
        sns.barplot(x="count", y="category", data=loc_cat, ax=ax4)
        st.pyplot(fig4)

        st.caption("Complaint types in this location")

    st.write("Total Reviews:", len(df_loc))
    st.write("Avg Rating:", round(df_loc["rating"].mean(), 2))

# =============================
# DRIVER ANALYSIS
# =============================
elif section == "Driver Analysis":

    st.subheader("🚖 Driver Performance Table")
    st.dataframe(driver_scores.head(10))

    st.subheader("🔥 Top High-Risk Drivers")

    fig, ax = plt.subplots()
    sns.barplot(
        x="risk",
        y="driver_id",
        data=driver_scores.sort_values("risk", ascending=False).head(5),
        ax=ax
    )
    st.pyplot(fig)

    selected_driver = st.selectbox("Select Driver", drivers)
    df_driver = fake_df[fake_df["driver_id"] == selected_driver]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df_driver['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        df_driver['risk'].value_counts().plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

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
        ### 🔍 Fake Detection Criteria

        • Very short reviews  
        • Repetitive or duplicate wording  
        • Unusual writing patterns  
        • Excess punctuation / uppercase / numbers  

        ### 🧠 Prediction Classes

        • Genuine → Real user feedback  
        • Fake → Suspicious / spam-like review  
        • Ambiguous → Uncertain case (borderline)  
        """)

    st.subheader("📋 Flagged Reviews")
    st.dataframe(fake_df.head(20))

# =============================
# SUBMIT REVIEW
# =============================
elif section == "Submit Review":

    st.subheader("📝 Submit Review")

    review = st.text_area("Your Review")
    rating = st.slider("Rating", 1, 5)
    location = st.selectbox("Location", locations)
    driver_id = st.selectbox("Driver", drivers)

    if st.button("Submit"):

        if review.strip() == "":
            st.warning("Please enter a review")
        else:
            st.success("Review submitted!")

            # -----------------------------
            # LOAD MODEL
            # -----------------------------
            

            # -----------------------------
            # FEATURE EXTRACTION
            # -----------------------------
            review_len = len(review.split())
            char_len = len(review)
            unique_words = len(set(review.split()))

            avg_word_len = np.mean([len(w) for w in review.split()]) if review.split() else 0

            repetition_ratio = review_len / (unique_words + 1)

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
            

            # -----------------------------
            # SCALE + PREDICT
            # -----------------------------
            features_scaled = fake_scaler.transform(features)
            score = fake_model.decision_function(features_scaled)[0]

            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            # -----------------------------
            # SMART HYBRID CLASSIFICATION
        # -----------------------------

# Rule-based overrides (VERY IMPORTANT)
            if repetition_ratio > 1.8:
                fake_label = "Fake"

            elif review_len <= 2:
                fake_label = "Fake"

            elif punctuation_count > 4:
                fake_label = "Fake"

# ML decision
            elif score < -0.15:
                fake_label = "Fake"

            elif score < 0.05:
                fake_label = "Ambiguous"

            else:
                fake_label = "Genuine"

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.subheader("🤖 Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Authenticity", fake_label)

            with col2:
                confidence = round(abs(score), 3)
                st.metric("Confidence Score", confidence)

            st.markdown("### 🔍 Why this prediction?")

            reasons = []
            if repetition_ratio > 1.8:
                reasons.append("High repetition in words")

            if repetition_ratio < 0.5:
                reasons.append("Low word diversity")

            if uppercase_ratio > 0.3:
                reasons.append("Too many uppercase characters")

            if punctuation_count > 3:
                reasons.append("Excessive punctuation")

            if digit_ratio > 0.2:
                reasons.append("Contains unusual numeric patterns")

            if review_len < 3:
                reasons.append("Very short review")

            if not reasons:
                reasons.append("Text looks natural")

            for r in reasons:
                st.write("•", r)
            st.caption("Higher confidence score = stronger model decision")

        
