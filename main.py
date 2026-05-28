"""
Student Performance Analysis - Main Script
Runs data cleaning, EDA visualizations, and optional ML model.
Execute from project root: python main.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "students.csv"
CHARTS_DIR = PROJECT_ROOT / "images" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)

SUBJECT_COLS = ["math_score", "science_score", "english_score", "history_score"]


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV dataset from data folder."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, column names, and dtypes."""
    df = df.copy()

    # Standardize column names (lowercase, underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate row(s)")

    # Numeric columns: coerce and fill missing with median
    numeric_cols = [
        "study_hours_per_day",
        "attendance_percent",
        *SUBJECT_COLS,
        "final_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing '{col}' with median: {median_val:.2f}")

    # Categorical: fill gender if missing
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown").astype(str)

    print("\nData types after cleaning:")
    print(df.dtypes)
    return df


def save_figure(name: str) -> None:
    """Save current matplotlib figure to images/charts."""
    path = CHARTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {path.relative_to(PROJECT_ROOT)}")


def run_eda_and_charts(df: pd.DataFrame) -> None:
    """Generate all required visualizations and save to images/charts."""

    # 1. Study hours vs final score (scatter)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="study_hours_per_day",
        y="final_score",
        hue="gender",
        alpha=0.75,
        s=80,
    )
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Final Score")
    plt.legend(title="Gender")
    save_figure("01_study_hours_vs_final_score.png")

    # 2. Attendance impact (scatter + trend)
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="attendance_percent",
        y="final_score",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "crimson"},
    )
    plt.title("Attendance Impact on Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    save_figure("02_attendance_vs_final_score.png")

    # 3. Gender-based performance (box plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="gender", y="final_score", hue="gender", legend=False)
    plt.title("Gender-Based Performance Comparison")
    plt.xlabel("Gender")
    plt.ylabel("Final Score")
    save_figure("03_gender_performance_boxplot.png")

    # 4. Subject-wise average (bar chart)
    subject_means = df[SUBJECT_COLS].mean()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ["Math", "Science", "English", "History"],
        subject_means.values,
        color=sns.color_palette("husl", 4),
        edgecolor="black",
    )
    plt.title("Subject-Wise Average Scores")
    plt.xlabel("Subject")
    plt.ylabel("Average Score")
    for bar, val in zip(bars, subject_means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center")
    save_figure("04_subject_wise_average.png")

    # 5. Correlation heatmap
    corr_cols = ["study_hours_per_day", "attendance_percent", *SUBJECT_COLS, "final_score"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap")
    save_figure("05_correlation_heatmap.png")

    # 6. Distribution of final scores (histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["final_score"], kde=True, bins=20, color="steelblue")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Frequency")
    save_figure("06_final_score_distribution.png")

    # 7. Gender distribution (pie chart)
    plt.figure(figsize=(8, 8))
    gender_counts = df["gender"].value_counts()
    plt.pie(
        gender_counts.values,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Student Gender Distribution")
    save_figure("07_gender_distribution_pie.png")

    # 8. Top 10 students (bar chart)
    top10 = df.nlargest(10, "final_score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x="final_score", y="student_name", hue="student_name", legend=False)
    plt.title("Top 10 Performing Students")
    plt.xlabel("Final Score")
    plt.ylabel("Student Name")
    save_figure("08_top_10_students.png")


def run_ml(df: pd.DataFrame) -> dict:
    """Linear regression: predict final_score from study hours and attendance."""
    X = df[["study_hours_per_day", "attendance_percent"]]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scatter plot: predicted vs actual (test set)
    plt.figure(figsize=(8, 8))
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.75,
        color="teal",
        edgecolors="black",
        linewidth=0.5,
        label="Predictions",
    )
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.title("Predicted vs Actual Final Score (Test Set)")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.legend()
    save_figure("09_predicted_vs_actual.png")

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return metrics


def print_summary(df: pd.DataFrame, metrics: dict) -> None:
    """Print key findings and final summary report."""
    subject_means = df[SUBJECT_COLS].mean()
    best_subject = subject_means.idxmax().replace("_score", "").title()

    corr_study = df["study_hours_per_day"].corr(df["final_score"])
    corr_attend = df["attendance_percent"].corr(df["final_score"])

    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    print(f"Total students (after cleaning): {len(df)}")
    print(f"Average final score: {df['final_score'].mean():.2f}")
    print(f"Best-performing subject: {best_subject} ({subject_means.max():.2f})")
    print(f"Study hours vs final score correlation: {corr_study:.3f}")
    print(f"Attendance vs final score correlation: {corr_attend:.3f}")
    print("\nTop 5 students:")
    print(
        df.nlargest(5, "final_score")[["student_name", "final_score", "study_hours_per_day", "attendance_percent"]]
        .to_string(index=False)
    )
    print("\nMachine Learning Metrics (Linear Regression):")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R2 Score: {metrics['R2']:.4f}")
    print("=" * 60)


def main() -> None:
    """Orchestrate load, clean, visualize, model, and report."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df_raw = load_data(DATA_PATH)
    df = clean_data(df_raw)
    run_eda_and_charts(df)
    metrics = run_ml(df)
    print_summary(df, metrics)


if __name__ == "__main__":
    main()
