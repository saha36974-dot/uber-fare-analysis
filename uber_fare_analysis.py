# 1. IMPORTING RELEVANT LIBRARIES
# Pandas and NumPy for handling data, Matplotlib and Seaborn for plotting,
# Scikit-learn for machine learning.
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set visualization style
sns.set_theme(style="whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze an Uber-style ride dataset.")
    parser.add_argument("csv_path", nargs="?", default="uber_data.csv", help="Path to the dataset CSV file.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where generated charts should be saved.",
    )
    return parser.parse_args()


def normalize_columns(df):
    alias_map = {
        "pickup_datetime": ["pickup_datetime", "tpep_pickup_datetime", "pickup_date", "timestamp"],
        "fare_amount": ["fare_amount", "fare", "total_fare"],
        "pickup_longitude": ["pickup_longitude", "pickup_x", "pickup_lon", "pickup_lng"],
        "pickup_latitude": ["pickup_latitude", "pickup_y", "pickup_lat"],
        "dropoff_longitude": ["dropoff_longitude", "dropoff_x", "dropoff_lon", "dropoff_lng"],
        "dropoff_latitude": ["dropoff_latitude", "dropoff_y", "dropoff_lat"],
    }

    for canonical, variants in alias_map.items():
        match = next((name for name in variants if name in df.columns), None)
        if match is not None and canonical not in df.columns:
            df = df.rename(columns={match: canonical})
    return df


def load_dataset(path):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    return df


def prepare_data(df):
    df = df.copy()
    df = df.dropna(subset=[col for col in df.columns if col in {"fare_amount"}])

    datetime_column = next((col for col in ["pickup_datetime"] if col in df.columns), None)
    if datetime_column is not None:
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
        df = df.dropna(subset=[datetime_column])
        df["hour"] = df[datetime_column].dt.hour
        df["day_of_week"] = df[datetime_column].dt.day_name()
        df["month"] = df[datetime_column].dt.month
        df["is_weekend"] = (df[datetime_column].dt.day_of_week >= 5).astype(int)

    if "fare_amount" in df.columns:
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200)].copy()

    return df


def save_plot(fig, filename, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / filename, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(df, output_dir):
    print("Generating visualizations...")

    if "hour" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x="hour", color="royalblue", ax=ax)
        ax.set_title("Uber Ride Demand by Hour of the Day")
        ax.set_xlabel("Hour (0-23)")
        ax.set_ylabel("Total Rides")
        save_plot(fig, "peak_hours.png", output_dir)

    if "day_of_week" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sns.countplot(data=df, x="day_of_week", order=order, color="coral", ax=ax)
        ax.set_title("Uber Ride Demand by Day of the Week")
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Total Rides")
        save_plot(fig, "rides_by_day.png", output_dir)

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        save_plot(fig, "correlation_heatmap.png", output_dir)


def train_model(df):
    print("Training predictive model...")

    target = "fare_amount"
    features = [
        "hour",
        "month",
        "is_weekend",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        print(f"Missing required columns for ML model: {', '.join(missing)}")
        return

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("Loading dataset...")
    df = load_dataset(args.csv_path)

    print("Cleaning data...")
    df = prepare_data(df)

    if df.empty:
        raise ValueError("The dataset is empty after cleaning. Check the input file and column names.")

    generate_visualizations(df, output_dir)
    train_model(df)


if __name__ == "__main__":
    main()