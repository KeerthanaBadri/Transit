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

    # -----------------------------
    # TEST BUTTONS 🔥
    # -----------------------------
    st.markdown("### 🧪 Quick Test Examples")

    colA, colB, colC = st.columns(3)

    if colA.button("Fake Example"):
        st.session_state.review = "good good good good driver driver best best amazing amazing"

    if colB.button("Genuine Example"):
        st.session_state.review = "The driver was polite and the ride was smooth and comfortable"

    if colC.button("Ambiguous Example"):
        st.session_state.review = "Ride was okay, nothing special but not bad either"

    # -----------------------------
    # INPUT
    # -----------------------------
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
            # FEATURE ENGINEERING
            # -----------------------------
            review_len = len(review.split())
            char_len = len(review)
            unique_words = len(set(review.split()))

            avg_word_len = np.mean([len(w) for w in review.split()]) if review.split() else 0

            repetition_ratio = review_len / (unique_words + 1)
            uppercase_ratio = sum(1 for c in review if c.isupper()) / (len(review) + 1)
            punctuation_count = len(re.findall(r'[!?.]', review))
            digit_ratio = sum(c.isdigit() for c in review) / (len(review) + 1)

            # -----------------------------
            # FIX: DataFrame (no warning)
            # -----------------------------
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
            # HYBRID CLASSIFICATION
            # -----------------------------
            if repetition_ratio > 1.8:
                label = "Fake"
            elif review_len <= 2:
                label = "Fake"
            elif punctuation_count > 4:
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
            st.subheader("🤖 Prediction Result")

            col1, col2 = st.columns(2)

            # COLOR OUTPUT
            with col1:
                if label == "Fake":
                    st.error(f"Authenticity: {label}")
                elif label == "Ambiguous":
                    st.warning(f"Authenticity: {label}")
                else:
                    st.success(f"Authenticity: {label}")

            # CONFIDENCE
            with col2:
                confidence = round(abs(score), 3)
                st.metric("Confidence Score", confidence)

            # PROGRESS BAR
            st.progress(min(abs(score), 1.0))

            # -----------------------------
            # EXPLANATION
            # -----------------------------
            st.markdown("### 🔍 Model Explanation")

            st.write(f"- Repetition Ratio: {round(repetition_ratio,2)}")
            st.write(f"- Uppercase Ratio: {round(uppercase_ratio,2)}")
            st.write(f"- Punctuation Count: {punctuation_count}")
            st.write(f"- Digit Ratio: {round(digit_ratio,2)}")

            st.markdown("### 🧠 Why this prediction?")

            reasons = []

            if repetition_ratio > 1.8:
                reasons.append("High repetition detected")

            if punctuation_count > 3:
                reasons.append("Excess punctuation")

            if uppercase_ratio > 0.3:
                reasons.append("Too many uppercase letters")

            if not reasons:
                reasons.append("Text appears natural")

            for r in reasons:
                st.write("•", r)