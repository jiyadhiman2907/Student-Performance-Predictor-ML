
import joblib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student_data_cleaned.csv"
DB_PATH = BASE_DIR / "predictions.db"


# Initialize SQLite database
def initialize_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_inputs TEXT NOT NULL,
                predicted_score REAL NOT NULL
            )
        """)


# Save a prediction to the database
def log_prediction(student_inputs, predicted_score):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                timestamp,
                student_inputs,
                predicted_score
            )
            VALUES (?, ?, ?)
            """,
            (
                timestamp,
                json.dumps(student_inputs),
                predicted_score
            )
        )


initialize_database()

# Load model and dataset
saved = joblib.load(MODEL_PATH)
model = saved["model"]
columns = saved["columns"]
df = pd.read_csv(DATA_PATH)

# Application heading
st.title("Student Performance Predictor")
st.write(
    "Enter a student's details to estimate their exam score. "
    "This tool supports self-assessment and is not an official grade."
)

# Student inputs
levels = ["Low", "Medium", "High"]

left, right = st.columns(2)

with left:
    hours = st.slider("Hours studied per week", 1, 44, 20)
    attendance = st.slider("Attendance (%)", 60, 100, 80)
    sleep = st.slider("Sleep hours per night", 4, 10, 7)
    previous = st.slider("Previous scores", 50, 100, 75)
    tutoring = st.slider("Tutoring sessions per month", 0, 8, 1)
    activity = st.slider(
        "Physical activity hours per week", 0, 6, 3
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    school = st.selectbox("School type", ["Public", "Private"])
    disability = st.selectbox(
        "Learning disabilities", ["No", "Yes"]
    )

with right:
    involvement = st.selectbox(
        "Parental involvement", levels, index=1
    )
    resources = st.selectbox(
        "Access to resources", levels, index=1
    )
    motivation = st.selectbox(
        "Motivation level", levels, index=1
    )
    income = st.selectbox("Family income", levels, index=1)
    teacher = st.selectbox("Teacher quality", levels, index=1)
    peer = st.selectbox(
        "Peer influence",
        ["Negative", "Neutral", "Positive"],
        index=1
    )
    parent_edu = st.selectbox(
        "Parental education level",
        ["High School", "College", "Postgraduate"]
    )
    distance = st.selectbox(
        "Distance from home", ["Near", "Moderate", "Far"]
    )
    extra = st.selectbox(
        "Extracurricular activities", ["No", "Yes"], index=1
    )
    internet = st.selectbox("Internet access", ["Yes", "No"])

# Generate prediction
if st.button("Predict exam score", type="primary"):
    student_inputs = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Parental_Involvement": involvement,
        "Access_to_Resources": resources,
        "Extracurricular_Activities": extra,
        "Sleep_Hours": sleep,
        "Previous_Scores": previous,
        "Motivation_Level": motivation,
        "Internet_Access": internet,
        "Tutoring_Sessions": tutoring,
        "Family_Income": income,
        "Teacher_Quality": teacher,
        "School_Type": school,
        "Peer_Influence": peer,
        "Physical_Activity": activity,
        "Learning_Disabilities": disability,
        "Parental_Education_Level": parent_edu,
        "Distance_from_Home": distance,
        "Gender": gender
    }

    row = pd.DataFrame([student_inputs])

    encoded = pd.get_dummies(row).reindex(
        columns=columns,
        fill_value=0
    )

    score = float(model.predict(encoded)[0])
    score = max(0.0, min(100.0, score))

    log_prediction(student_inputs, score)

    st.subheader(f"Predicted exam score: {score:.1f} / 100")
    st.success("Prediction saved to the database.")

    if score >= 70:
        st.info("Expected performance: strong")
    elif score >= 65:
        st.info("Expected performance: average")
    else:
        st.warning(
            "Expected performance: below average. "
            "Additional academic support may help."
        )

    # Compare prediction with dataset statistics
    st.subheader("Your Score Compared with the Dataset")

    average_score = df["Exam_Score"].mean()
    median_score = df["Exam_Score"].median()

    comparison = pd.DataFrame({
        "Measure": [
            "Your predicted score",
            "Dataset average",
            "Dataset median"
        ],
        "Score": [score, average_score, median_score]
    })

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        comparison["Measure"],
        comparison["Score"],
        edgecolor="black"
    )

    ax.set_ylabel("Exam Score")
    ax.set_ylim(0, 100)
    ax.set_title("Predicted Score vs Dataset Statistics")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}",
            ha="center"
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Prediction insights: compare the estimate with observed dataset scores
    st.subheader("Prediction Insights")

    score_difference = score - average_score
    dataset_percentile = (df["Exam_Score"] <= score).mean() * 100

    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        st.metric(
            "Difference from Dataset Average",
            f"{score_difference:+.1f} points"
        )

    with insight2:
        st.metric(
            "Position in Dataset Score Distribution",
            f"{dataset_percentile:.0f}th percentile"
        )

    with insight3:
        st.metric(
            "Dataset Median",
            f"{median_score:.1f} / 100"
        )

    if score_difference > 0.5:
        st.info(
            "The predicted score is above the dataset average. "
            "This is a comparison with the dataset, not a guarantee "
            "of actual exam performance."
        )
    elif score_difference < -0.5:
        st.info(
            "The predicted score is below the dataset average. "
            "Consider reviewing study habits and seeking academic support "
            "where useful. This estimate is not an official grade."
        )
    else:
        st.info(
            "The predicted score is close to the dataset average. "
            "Actual results may differ from this estimate."
        )

    st.caption(
        "Percentile is calculated by comparing the predicted score with "
        "the observed Exam_Score values in the loaded dataset. It describes "
        "relative position in this dataset only; it is not a rank among "
        "all students or a prediction of future class standing."
    )

# Student performance dashboard
st.divider()
st.header("Student Performance Dashboard")

# Chart 1: Exam score distribution
st.subheader("Exam Score Distribution")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["Exam_Score"], bins=20, edgecolor="black")
ax.set_xlabel("Exam Score")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Exam Scores")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Chart 2: Numerical feature correlations
st.subheader("Feature Correlation with Exam Score")

numeric_columns = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score"
]

correlations = (
    df[numeric_columns]
    .corr()["Exam_Score"]
    .drop("Exam_Score")
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 4))
correlations.plot(kind="barh", ax=ax, edgecolor="black")
ax.set_xlabel("Correlation with Exam Score")
ax.set_ylabel("Feature")
ax.set_title("Numerical Feature Correlations")
ax.axvline(0, linewidth=1)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Model evaluation
st.divider()
st.header("Model Evaluation")

st.write(
    "The following metrics are from the recorded test-set "
    "evaluation performed during model training."
)

evaluation = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Decision Tree"
    ],
    "MAE": [0.452, 1.184, 1.600],
    "RMSE": [1.804, 2.233, 2.511],
    "R2 Score": [0.770, 0.647, 0.554]
})

st.success(
    "Selected model: Linear Regression. It achieved the best "
    "recorded test-set results among the three evaluated models."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("MAE", "0.452")

with metric2:
    st.metric("RMSE", "1.804")

with metric3:
    st.metric("R² Score", "0.770")

st.dataframe(
    evaluation,
    hide_index=True,
    use_container_width=True
)

with st.expander("Understand the evaluation metrics"):
    st.markdown("""
    **MAE (Mean Absolute Error)**

    The average absolute difference between predicted and actual scores.
    Lower is better.

    **RMSE (Root Mean Squared Error)**

    Measures prediction errors while penalizing larger errors more strongly.
    Lower is better.

    **R² Score (Coefficient of Determination)**

    Measures how much variation in the target is explained by the model
    on the evaluated dataset. Higher is generally better.

    **Important:** These are recorded test-set results and do not guarantee
    accuracy for every new student.
    """)

# Prediction history and summary
st.divider()
st.header("Prediction History")

with sqlite3.connect(DB_PATH) as connection:
    history = pd.read_sql_query(
        """
        SELECT id, timestamp, predicted_score
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
        """,
        connection
    )

    all_predictions = pd.read_sql_query(
        """
        SELECT id, timestamp, student_inputs, predicted_score
        FROM predictions
        ORDER BY id DESC
        """,
        connection
    )

    summary = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_predictions,
            AVG(predicted_score) AS average_score,
            MAX(predicted_score) AS highest_score
        FROM predictions
        """,
        connection
    )

total_predictions = int(summary.loc[0, "total_predictions"])
average_score = summary.loc[0, "average_score"]
highest_score = summary.loc[0, "highest_score"]

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Total Predictions", total_predictions)

with metric2:
    st.metric(
        "Average Predicted Score",
        f"{average_score:.1f}" if pd.notna(average_score) else "N/A"
    )

with metric3:
    st.metric(
        "Highest Predicted Score",
        f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A"
    )

if history.empty:
    st.info("No predictions have been recorded yet.")
else:
    st.caption("Showing the 10 most recent predictions.")

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True
    )

# Download all prediction records
st.subheader("Export Prediction History")

if all_predictions.empty:
    st.info("Make a prediction before downloading history.")
else:
    csv_data = all_predictions.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History (CSV)",
        data=csv_data,
        file_name="student_prediction_history.csv",
        mime="text/csv"
    )

# Disclaimer
st.caption(
    "This is a decision-support and self-assessment tool, "
    "not an official grade. Predictions are estimates and "
    "are not guaranteed exam results."
)
