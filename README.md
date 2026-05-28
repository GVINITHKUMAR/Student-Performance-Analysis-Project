# Student Performance Analysis

A complete **Exploratory Data Analysis (EDA)** and optional **machine learning** project that analyzes student academic performance using Python, pandas, and Jupyter Notebook. Built for learning, portfolios, and GitHub.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview

This project investigates how **study hours**, **attendance**, **gender**, and **subject scores** relate to overall student performance. It includes:

- Data loading and cleaning (missing values, duplicates, dtypes)
- Full EDA with statistical summaries
- Professional visualizations (bar, scatter, histogram, heatmap, box, pie)
- Linear Regression to predict final scores
- A printable final summary report

---

## Dataset Details

**File:** `data/students.csv`

| Column | Description |
|--------|-------------|
| `student_id` | Unique student identifier |
| `student_name` | Student name |
| `gender` | Male / Female / Other |
| `study_hours_per_day` | Average daily study hours |
| `attendance_percent` | Class attendance percentage |
| `math_score` | Math exam score (0–100) |
| `science_score` | Science exam score (0–100) |
| `english_score` | English exam score (0–100) |
| `history_score` | History exam score (0–100) |
| `final_score` | Overall average score |

- **Records:** 120 students (+ intentional duplicates and missing values for cleaning practice)
- **Source:** Synthetic dataset generated for educational purposes

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical operations |
| **matplotlib** | Plotting and chart export |
| **seaborn** | Statistical visualizations |
| **scikit-learn** | Linear Regression model |
| **Jupyter Notebook** | Interactive analysis |

---

## Project Structure

```
Student-Performance-Analysis/
│
├── data/
│   └── students.csv          # Dataset
│
├── notebooks/
│   └── analysis.ipynb        # Main EDA notebook
│
├── images/
│   └── charts/               # Saved visualization outputs
│
├── README.md
├── requirements.txt
├── main.py                   # CLI script (charts + ML + report)
└── generate_data.py          # Optional: regenerate CSV
```

---

## Steps to Run the Project

### 1. Clone or download the repository

```bash
git clone <your-repo-url>
cd Student-Performance-Analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook (recommended for learning)

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells from top to bottom. Charts are saved to `images/charts/`.

### 5. Or run the standalone script

```bash
python main.py
```

This loads data, cleans it, generates all charts, trains the ML model, and prints the summary report.

### 6. Regenerate dataset (optional)

```bash
python generate_data.py
```

---

## Sample Outputs

After running the notebook or `main.py`, you will find charts in `images/charts/`:

| Chart | Filename |
|-------|----------|
| Study hours vs final score | `01_study_hours_vs_final_score.png` |
| Attendance impact | `02_attendance_vs_final_score.png` |
| Gender performance (box plot) | `03_gender_performance_boxplot.png` |
| Subject-wise averages | `04_subject_wise_average.png` |
| Correlation heatmap | `05_correlation_heatmap.png` |
| Score distribution | `06_final_score_distribution.png` |
| Gender pie chart | `07_gender_distribution_pie.png` |
| Top 10 students | `08_top_10_students.png` |
| ML: Predicted vs actual | `09_predicted_vs_actual.png` |

**Example console output (ML metrics):**

```
Machine Learning Metrics (Linear Regression):
  MAE: 3.4521
  MSE: 18.2341
  R2 Score: 0.7823
```

*(Exact values may vary slightly after data cleaning.)*

---

## Insights

1. **Study hours and marks:** A positive correlation exists between daily study hours and final score. Students who study more tend to score higher.

2. **Attendance impact:** Higher attendance strongly associates with better final scores. Students below 70% attendance average noticeably lower marks.

3. **Best-performing subject:** Typically **Math** or **English** leads in average scores (verify in your run via the summary report).

4. **Gender comparison:** Average performance across gender groups is relatively similar; individual study habits and attendance matter more.

5. **Correlation:** Study hours and attendance both correlate positively with `final_score` and subject scores.

6. **Machine learning:** Linear Regression using study hours and attendance achieves reasonable **R²** on the test set, useful for early risk identification.

7. **Distribution:** Final scores are approximately normal with most students clustered around the class mean.

---

## Resume / Portfolio Tips

- Link this repo on your resume under **Projects**
- Mention: EDA, data cleaning, visualization, scikit-learn, Jupyter
- Include 1–2 chart screenshots in your README or portfolio site

---

## License

MIT License — free to use for learning and portfolio projects.

---

## Author

Created as an educational data analytics project. Contributions and feedback welcome via issues and pull requests.
