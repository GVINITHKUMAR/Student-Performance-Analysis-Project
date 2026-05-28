"""Generate students.csv with realistic synthetic data (run once)."""
import csv
import random

random.seed(42)

ROWS = []
for i in range(1, 121):
    gender = random.choices(["Male", "Female", "Other"], weights=[48, 48, 4])[0]
    study = round(max(1, min(12, random.gauss(5.5, 2.2))), 1)
    attend = round(max(50, min(100, random.gauss(82, 12))), 1)
    math = round(max(0, min(100, 40 + study * 4 + attend * 0.25 + random.gauss(0, 8))), 1)
    science = round(max(0, min(100, 38 + study * 3.8 + attend * 0.28 + random.gauss(0, 9))), 1)
    english = round(max(0, min(100, 42 + study * 3.5 + attend * 0.22 + random.gauss(0, 10))), 1)
    history = round(max(0, min(100, 39 + study * 3.2 + attend * 0.26 + random.gauss(0, 9))), 1)
    final = round((math + science + english + history) / 4, 1)
    ROWS.append(
        [i, f"Student_{i:03d}", gender, study, attend, math, science, english, history, final]
    )

# Missing values
for idx in random.sample(range(120), 8):
    ROWS[idx][4] = ""
for idx in random.sample(range(120), 6):
    ROWS[idx][6] = ""
for idx in random.sample(range(120), 4):
    ROWS[idx][3] = ""

# Duplicate rows
ROWS.extend(ROWS[:3])

HEADER = [
    "student_id",
    "student_name",
    "gender",
    "study_hours_per_day",
    "attendance_percent",
    "math_score",
    "science_score",
    "english_score",
    "history_score",
    "final_score",
]

with open("data/students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(ROWS)

print(f"Wrote {len(ROWS)} rows to data/students.csv")
