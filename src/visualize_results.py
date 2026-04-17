import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -----------------------------
# Load both results
# -----------------------------
sentiment_df = pd.read_csv("data/processed/sentiment_results.csv")
risk_df = pd.read_csv("data/processed/risk_results.csv")

# Add type column
sentiment_df["Task"] = "Sentiment"
risk_df["Task"] = "Risk"

# Combine
df = pd.concat([sentiment_df, risk_df], ignore_index=True)

print(df)

# -----------------------------
# Plot function
# -----------------------------
def plot_metric(metric):
    plt.figure()
    sns.barplot(x="Model", y=metric, hue="Task", data=df)
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Generate all graphs
# -----------------------------
plot_metric("Accuracy")
plot_metric("Precision")
plot_metric("Recall")
plot_metric("F1 Score")