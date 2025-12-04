"""
analysis.py
A complete single-file solution for data sourcing, preparation, analysis,
visualization, and LLM-based insights for the OpenAI LLM Analysis Quiz project.
"""

EMAIL = "ramyaiitm@gmail.com"
print("Student Email:", EMAIL)

# --------------------------
# IMPORTS
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openai import OpenAI

# --------------------------
# CONFIG
# --------------------------
DATA_FILE = "data.csv"        # put your dataset file name
PLOT_FILE = "plot.png"        # visualization output
MODEL = "gpt-4o-mini"         # or any allowed model
client = OpenAI()

# --------------------------
# 1. LOAD DATA
# --------------------------
def load_data(file_path=DATA_FILE):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(df.head())
    return df

# --------------------------
# 2. CLEAN DATA
# --------------------------
def clean_data(df):
    print("\nCleaning data...")
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    print("Data after cleaning:")
    print(df.head())
    return df

# --------------------------
# 3. DATA ANALYSIS
# --------------------------
def basic_analysis(df):
    print("\nBasic Statistical Summary:")
    print(df.describe())

    print("\nCorrelation Matrix:")
    print(df.corr())

# --------------------------
# 4. K-Means Clustering
# --------------------------
def clustering(df, k=3):
    print("\nRunning K-Means clustering...")

    numeric_df = df.select_dtypes(include=["number"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled)

    df["Cluster"] = labels
    print("Clusters assigned.")
    print(df.head())
    return df

# --------------------------
# 5. VISUALIZATION
# --------------------------
def create_plot(df, x_col, y_col):
    print(f"\nGenerating plot ({x_col} vs {y_col})...")

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"Plot saved as {PLOT_FILE}")

# --------------------------
# 6. LLM INSIGHTS
# --------------------------
def generate_insights(df):
    print("\nQuerying LLM for insights...")

    summary_text = df.describe().to_string()

    prompt = f"""
You are a data analyst. Based on the statistical summary below,
give meaningful insights and patterns from the data:

STATISTICS:
{summary_text}

Explain in bullet points.
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    insights = completion.choices[0].message["content"]
    print("\nLLM Insights:\n")
    print(insights)

    return insights

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    basic_analysis(df)
    df = clustering(df, k=3)

    # choose any two numeric columns for plotting
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 2:
        create_plot(df, numeric_cols[0], numeric_cols[1])
    else:
        print("Not enough numeric columns for plotting.")

    generate_insights(df)

    print("\n--- Analysis Completed Successfully ---")
