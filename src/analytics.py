import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/fake_detection_results.csv")
print("Data Loaded ✅")

# -----------------------------
# FIX CATEGORY LABELS (ROBUST)
# -----------------------------
category_map = {
    0: "General",
    1: "Driver Behavior",
    2: "Delay",
    3: "Safety",
    4: "Vehicle Condition",
    5: "Pricing"
}

# Convert to numeric safely, then map
df["category"] = pd.to_numeric(df["category"], errors="coerce")
df["category"] = df["category"].map(category_map)
df["category"] = df["category"].fillna("Other")

# -----------------------------
# 1. PEAK HOURS
# -----------------------------
peak_hours = (
    df.groupby("hour")
    .size()
    .reset_index(name="review_count")
    .sort_values("hour")
)

# -----------------------------
# 2. HIGH RISK LOCATIONS (TOP 10)
# -----------------------------
# If risk is numeric (0,1,2), use 2 for high risk
# If risk is string, fallback handled below
if df["risk"].dtype == "object":
    high_risk_df = df[df["risk"].str.lower() == "high"]
else:
    high_risk_df = df[df["risk"] == 2]

location_risk = (
    high_risk_df["location"]
    .value_counts()
    .head(10)
    .reset_index()
)

location_risk.columns = ["location", "high_risk_count"]

# -----------------------------
# 3. DRIVER PERFORMANCE
# -----------------------------
driver_scores = (
    df.groupby("driver_id")
    .agg({
        "risk": "mean",
        "rating": "mean",
        "review": "count"
    })
    .reset_index()
)

driver_scores.rename(columns={"review": "total_reviews"}, inplace=True)

# -----------------------------
# 4. TREND (TIME-BASED)
# -----------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")

trend = (
    df.dropna(subset=["time"])
    .groupby(df["time"].dt.date)
    .size()
    .reset_index(name="reviews")
)

trend.columns = ["time", "reviews"]

# -----------------------------
# 5. CATEGORY DISTRIBUTION (FIXED)
# -----------------------------
category_counts = (
    df["category"]
    .value_counts()
    .reset_index()
)

category_counts.columns = ["category", "count"]

# -----------------------------
# SAVE FILES
# -----------------------------
peak_hours.to_csv("data/processed/peak_hours.csv", index=False)
location_risk.to_csv("data/processed/location_risk.csv", index=False)
driver_scores.to_csv("data/processed/driver_scores.csv", index=False)
trend.to_csv("data/processed/trend.csv", index=False)
category_counts.to_csv("data/processed/category.csv", index=False)

print("Analytics fixed & saved ✅")