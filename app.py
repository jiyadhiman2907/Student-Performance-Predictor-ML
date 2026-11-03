import joblib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PASTEL PINK + PURPLE THEME
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --pink: #F6DCEB;
    --pink-soft: #FFF5FA;
    --lavender: #E9DDF8;
    --purple: #A78BCB;
    --purple-dark: #72538F;
    --plum: #49345F;
    --ink: #392D46;
    --muted: #81758C;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(246,220,235,.62), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(233,221,248,.70), transparent 30%),
        #FFFBFE;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8EAF3 0%, #F0E8FA 100%);
    border-right: 1px solid #E9DDF1;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #59476B; }
.block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3 { color: var(--plum) !important; letter-spacing: -0.035em; }
h1 { font-family: 'Playfair Display', serif !important; font-size: clamp(2.2rem, 4vw, 3.25rem) !important; }
h2 { font-size: 1.65rem !important; }
h3 { font-size: 1.2rem !important; }
p, label, span { color: var(--ink); }
[data-testid="stCaptionContainer"] p { color: var(--muted); }
.hero {
    padding: 2rem 2.2rem;
    border-radius: 26px;
    background: linear-gradient(115deg, #F9E5F0 0%, #F1E8FB 58%, #E8DDF7 100%);
    border: 1px solid #EAD9F0;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 35px rgba(91, 62, 117, .07);
}
.hero-kicker {
    color: #86689E; text-transform: uppercase; letter-spacing: .16em;
    font-size: .72rem; font-weight: 800; margin-bottom: .65rem;
}
.hero-title {
    font-family: 'Playfair Display', serif; color: #49345F;
    font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12; margin: 0 0 .65rem 0;
}
.hero-copy { color: #6D5C7C; font-size: 1rem; max-width: 760px; line-height: 1.65; }
.section-kicker {
    color: #9A7BB2; text-transform: uppercase; letter-spacing: .13em;
    font-size: .68rem; font-weight: 800; margin-bottom: .25rem;
}
.section-heading { font-family: 'Playfair Display', serif; font-size: 1.55rem; color: #49345F; margin: 0 0 .35rem 0; }
.section-copy { color: #81758C; margin-bottom: 1rem; font-size: .92rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.86); border: 1px solid #EDE3F2;
    border-radius: 18px; box-shadow: 0 5px 20px rgba(76, 51, 99, .045);
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF, #FFF7FB);
    border: 1px solid #EEDFEB; padding: 1rem 1.1rem; border-radius: 17px;
    box-shadow: 0 5px 16px rgba(76, 51, 99, .045);
}
div[data-testid="stMetricLabel"] p { color: #86758F !important; font-weight: 600; }
div[data-testid="stMetricValue"] { color: #5B3E78; font-weight: 800; }
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid #A78BCB;
    background: linear-gradient(110deg, #9E7ABD, #795A9A);
    color: white; font-weight: 700; padding: .62rem 1.2rem; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #72538F; background: #684982; color: white; transform: translateY(-1px);
}
.stButton > button:focus, .stDownloadButton > button:focus { box-shadow: 0 0 0 .2rem rgba(167,139,203,.25); }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    border-radius: 10px; border-color: #E5D8ED; background: #FFFEFF;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background-color: #8F6BAC; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #F7EFF9; padding: 7px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #725F82; font-weight: 700; }
.stTabs [aria-selected="true"] { background: white; color: #654781; }
hr { border-color: #EDE3F2 !important; }
div[data-testid="stAlert"] { border-radius: 13px; }
div[data-testid="stExpander"] { border: 1px solid #EDE3F2; border-radius: 13px; background: rgba(255,255,255,.72); }
div[data-testid="stDataFrame"] { border: 1px solid #EDE3F2; border-radius: 12px; overflow: hidden; }
.recommendation-number {
    display: inline-block; background: #F2E8FA; color: #72538F;
    border-radius: 9px; padding: .22rem .55rem; font-weight: 800; margin-right: .45rem;
}
.sidebar-brand { font-family: 'Playfair Display', serif; color: #553B70; font-size: 1.35rem; font-weight: 700; }
.sidebar-note { color: #806E8E; font-size: .82rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student_data_cleaned.csv"
DB_PATH = BASE_DIR / "predictions.db"


# ==================================================
# DATABASE FUNCTIONS
# ==================================================

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


def log_prediction(student_inputs, predicted_score):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """INSERT INTO predictions (timestamp, student_inputs, predicted_score)
               VALUES (?, ?, ?)""",
            (timestamp, json.dumps(student_inputs), predicted_score)
        )


# ==================================================
# LOAD MODEL AND DATASET
# ==================================================

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["columns"]


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


initialize_database()
model, columns = load_model()
df = load_dataset()


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_score(student_inputs):
    row = pd.DataFrame([student_inputs])
    encoded = pd.get_dummies(row).reindex(columns=columns, fill_value=0)
    score = float(model.predict(encoded)[0])
    return max(0.0, min(100.0, score))


# ==================================================
# PERSONALIZED STUDY RECOMMENDATIONS
# ==================================================

def generate_recommendations(student_inputs, predicted_score):
    recommendations = []

    rules = [
        ("Hours_Studied", lambda x: x < 10,
         "Build a Consistent Study Schedule",
         "Your reported study time is relatively low.",
         "Try setting aside regular study sessions during the week. Start with manageable sessions and increase them gradually."),
        ("Attendance", lambda x: x < 75,
         "Focus on Class Attendance",
         "Your reported attendance is below 75%.",
         "Try to attend classes regularly and review missed lessons to stay up to date."),
        ("Sleep_Hours", lambda x: x < 6,
         "Review Your Sleep Routine",
         "You reported fewer than 6 hours of sleep.",
         "Consider maintaining a regular sleep schedule and allowing sufficient time for rest."),
        ("Previous_Scores", lambda x: x < 65,
         "Strengthen Your Academic Fundamentals",
         "Your reported previous score is relatively low.",
         "Identify topics that you find difficult, revise the basics and practise questions regularly."),
        ("Tutoring_Sessions", lambda x: x == 0,
         "Explore Additional Academic Support",
         "You reported no tutoring sessions.",
         "If you are struggling with a subject, consider asking your teachers, classmates or mentors for help."),
        ("Motivation_Level", lambda x: x == "Low",
         "Set Small Academic Goals",
         "You reported a low motivation level.",
         "Break larger study tasks into smaller goals and track your progress after completing each one."),
        ("Access_to_Resources", lambda x: x == "Low",
         "Explore Learning Resources",
         "You reported limited access to learning resources.",
         "Explore library materials, free educational websites and resources provided by your institution."),
        ("Teacher_Quality", lambda x: x == "Low",
         "Seek Clarification and Guidance",
         "You reported low teacher quality.",
         "Consider discussing difficult topics with another teacher, mentor or academic support group."),
        ("Internet_Access", lambda x: x == "No",
         "Find Offline Learning Materials",
         "You reported that internet access is unavailable.",
         "Consider using textbooks, printed notes and offline study materials."),
        ("Peer_Influence", lambda x: x == "Negative",
         "Build a Supportive Study Environment",
         "You reported negative peer influence.",
         "Try studying with classmates who encourage productive learning habits.")
    ]

    for key, condition, title, reason, action in rules:
        if condition(student_inputs[key]):
            recommendations.append({
                "title": title, "reason": reason, "action": action
            })

    if not recommendations:
        recommendations.append({
            "title": "Maintain Your Current Study Habits",
            "reason": "Your submitted inputs did not trigger any of the specific recommendation rules.",
            "action": "Continue reviewing your progress, practising regularly and maintaining a balanced routine."
        })

    if predicted_score < 65:
        recommendations.insert(0, {
            "title": "Create an Academic Improvement Plan",
            "reason": "Your model-estimated score is below 65.",
            "action": "Identify your most challenging subjects, set weekly learning goals and seek academic support where needed."
        })

    return recommendations


# ==================================================
# PDF REPORT GENERATION
# ==================================================

def create_pdf_report(student_inputs, predicted_score, average_score,
                      median_score, dataset_percentile, recommendations,
                      simulated_hours, simulated_attendance, simulated_score):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, leading=25,
        alignment=TA_CENTER, textColor=colors.HexColor("#654781"), spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=10,
        alignment=TA_CENTER, textColor=colors.HexColor("#806E8E"), spaceAfter=15
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13, leading=17,
        textColor=colors.HexColor("#654781"), spaceBefore=12, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=5
    )
    small_style = ParagraphStyle(
        "SmallText", parent=styles["BodyText"], fontSize=8, leading=11
    )

    story = [
        Paragraph("STUDENT PERFORMANCE REPORT", title_style),
        Paragraph("Machine Learning-Based Student Performance Predictor", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#B99BCF")),
        Spacer(1, 10),
        Paragraph(f"<b>Report generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", body_style),
        Paragraph("1. Prediction Result", section_style)
    ]

    result_table = [
        ["Measure", "Value"],
        ["Predicted Exam Score", f"{predicted_score:.2f} / 100"],
        ["Dataset Average", f"{average_score:.2f}"],
        ["Dataset Median", f"{median_score:.2f}"],
        ["Dataset Percentile", f"{dataset_percentile:.1f}th"]
    ]
    table = Table(result_table, colWidths=[75 * mm, 85 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#765994")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#E1D6E9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBF5FA")]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(table)

    story.append(Paragraph("2. Student Input Details", section_style))
    input_rows = [["Input Parameter", "Submitted Value"]]
    input_rows.extend([[key.replace("_", " "), str(value)] for key, value in student_inputs.items()])
    input_table = Table(input_rows, colWidths=[75 * mm, 85 * mm], repeatRows=1)
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#765994")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#E1D6E9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBF5FA")]),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.extend([input_table, Paragraph("3. Prediction Insights", section_style)])
    story.append(Paragraph(
        f"Difference from dataset average: <b>{predicted_score - average_score:+.2f} points</b>", body_style
    ))
    story.append(Paragraph(
        f"Position in dataset score distribution: <b>{dataset_percentile:.1f}th percentile</b>", body_style
    ))

    story.append(Paragraph("4. Personalized Study Recommendations", section_style))
    for index, recommendation in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{index}. {recommendation['title']}</b>", body_style))
        story.append(Paragraph(recommendation["reason"], small_style))
        story.append(Paragraph(f"<b>Suggested action:</b> {recommendation['action']}", small_style))
        story.append(Spacer(1, 5))

    story.append(Paragraph("5. What-If Performance Simulation", section_style))
    simulation_rows = [
        ["Measure", "Original", "Simulated"],
        ["Study hours per week", str(student_inputs["Hours_Studied"]), str(simulated_hours)],
        ["Attendance", f"{student_inputs['Attendance']}%", f"{simulated_attendance}%"],
        ["Predicted score", f"{predicted_score:.2f}", f"{simulated_score:.2f}"]
    ]
    simulation_table = Table(simulation_rows, colWidths=[65 * mm, 45 * mm, 50 * mm])
    simulation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#765994")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#E1D6E9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBF5FA")]),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.append(simulation_table)
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        f"Difference between original and simulated prediction: <b>{simulated_score - predicted_score:+.2f} points</b>",
        body_style
    ))

    story.append(Paragraph("6. Important Disclaimer", section_style))
    story.append(Paragraph(
        "This report contains machine learning-based estimates generated from the information submitted by the user. "
        "Predictions are not official examination grades and are not guaranteed results. Dataset comparisons describe "
        "relative position within the loaded dataset. Study recommendations are generated using predefined rules. "
        "What-if results are model-based estimates and do not establish cause and effect.", small_style
    ))
    story.extend([Spacer(1, 15), Paragraph("Generated by Student Performance Predictor", subtitle_style)])
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==================================================
# DISPLAY HELPERS
# ==================================================

def section_header(kicker, title, description=None):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>'
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if description:
        st.markdown(f'<div class="section-copy">{description}</div>', unsafe_allow_html=True)


def style_chart(ax):
    ax.set_facecolor("#FFFBFE")
    ax.tick_params(colors="#756681")
    ax.xaxis.label.set_color("#756681")
    ax.yaxis.label.set_color("#756681")
    ax.title.set_color("#533D69")
    for spine in ax.spines.values():
        spine.set_color("#E8DDED")
    ax.grid(axis="y", alpha=.18, color="#B99BCF")


# ==================================================
# SESSION STATE
# ==================================================

if "baseline_inputs" not in st.session_state:
    st.session_state.baseline_inputs = None
if "baseline_score" not in st.session_state:
    st.session_state.baseline_score = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Student Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Understand academic progress through data-driven insights.</div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("**EXPLORE**")
    st.markdown("🎓  Predict Performance")
    st.markdown("📚  Study Recommendations")
    st.markdown("🔍  What-If Analysis")
    st.markdown("📊  Performance Dashboard")
    st.divider()
    st.markdown("**PROJECT OVERVIEW**")
    st.markdown(
        '<div class="sidebar-note">A machine learning application that estimates student exam performance '
        'and helps identify factors that may support better academic outcomes.</div>',
        unsafe_allow_html=True
    )
    st.caption("Python · Machine Learning · Streamlit")


# ==================================================
# HERO
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">Your academic snapshot</div>
    <div class="hero-title">Understand your progress.<br>Plan your next step.</div>
    <div class="hero-copy">
        Explore a model-based estimate of exam performance, compare it with
        the dataset, and discover practical ways to plan your learning.
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# STUDENT INPUT FORM
# ==================================================

section_header(
    "01 · STUDENT PROFILE",
    "Tell us about your learning routine",
    "Adjust the details below, then generate your performance estimate."
)

levels = ["Low", "Medium", "High"]

with st.container(border=True):
    st.markdown("**Academic and lifestyle details**")
    left, right = st.columns(2)

    with left:
        hours = st.slider("Hours studied per week", 1, 44, 20)
        attendance = st.slider("Attendance (%)", 60, 100, 80)
        sleep = st.slider("Sleep hours per night", 4, 10, 7)
        previous = st.slider("Previous scores", 50, 100, 75)
        tutoring = st.slider("Tutoring sessions per month", 0, 8, 1)
        activity = st.slider("Physical activity hours per week", 0, 6, 3)
        gender = st.selectbox("Gender", ["Female", "Male"])
        school = st.selectbox("School type", ["Public", "Private"])
        disability = st.selectbox("Learning disabilities", ["No", "Yes"])

    with right:
        involvement = st.selectbox("Parental involvement", levels, index=1)
        resources = st.selectbox("Access to resources", levels, index=1)
        motivation = st.selectbox("Motivation level", levels, index=1)
        income = st.selectbox("Family income", levels, index=1)
        teacher = st.selectbox("Teacher quality", levels, index=1)
        peer = st.selectbox("Peer influence", ["Negative", "Neutral", "Positive"], index=1)
        parent_edu = st.selectbox("Parental education level", ["High School", "College", "Postgraduate"])
        distance = st.selectbox("Distance from home", ["Near", "Moderate", "Far"])
        extra = st.selectbox("Extracurricular activities", ["No", "Yes"], index=1)
        internet = st.selectbox("Internet access", ["Yes", "No"])

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

    predict_col, note_col = st.columns([1, 2])
    with predict_col:
        predict_clicked = st.button("✦  Generate prediction", type="primary", use_container_width=True)
    with note_col:
        st.caption("Your estimate is generated using the saved Linear Regression model.")

if predict_clicked:
    score = predict_score(student_inputs)
    st.session_state.baseline_inputs = student_inputs.copy()
    st.session_state.baseline_score = score
    log_prediction(student_inputs, score)


# ==================================================
# PREDICTION RESULTS
# ==================================================

if st.session_state.baseline_score is not None:
    score = st.session_state.baseline_score
    original_inputs = st.session_state.baseline_inputs
    average_score = float(df["Exam_Score"].mean())
    median_score = float(df["Exam_Score"].median())
    dataset_percentile = float((df["Exam_Score"] <= score).mean() * 100)
    recommendations = generate_recommendations(original_inputs, score)

    st.divider()
    section_header("02 · YOUR RESULTS", "Your performance snapshot",
                   "A model-generated estimate based on the details you submitted.")

    score_col, status_col = st.columns([1, 2])
    with score_col:
        st.markdown(
            f"""<div style="background:linear-gradient(145deg,#F8E4F0,#EDE2F8);
            border:1px solid #E6D5EE;border-radius:22px;padding:1.5rem;text-align:center;">
            <div style="color:#80658F;font-size:.76rem;font-weight:800;letter-spacing:.12em;">
            ESTIMATED EXAM SCORE</div>
            <div style="font-family:'Playfair Display',serif;color:#563C70;
            font-size:3.4rem;font-weight:700;line-height:1.2;">{score:.1f}</div>
            <div style="color:#80658F;font-size:.9rem;">out of 100</div></div>""",
            unsafe_allow_html=True
        )
    with status_col:
        st.markdown("#### Your result at a glance")
        if score >= 70:
            st.success("Strong estimated performance range")
        elif score >= 65:
            st.info("Average estimated performance range")
        else:
            st.warning("Below-average estimated range — additional academic support may help.")
        st.caption("This is a model estimate, not an official grade or a guarantee of actual examination results.")
        st.metric("Compared with dataset average", f"{score - average_score:+.1f} points")

    # Score comparison
    section_header("03 · DATASET COMPARISON", "How does your estimate compare?",
                   "Compare the prediction with the average and median score in the loaded dataset.")
    comparison = pd.DataFrame({
        "Measure": ["Your predicted score", "Dataset average", "Dataset median"],
        "Score": [score, average_score, median_score]
    })
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(comparison["Measure"], comparison["Score"],
                  color=["#A78BCB", "#E8BBD5", "#C9B5E2"], width=.58)
    ax.set_ylabel("Exam score")
    ax.set_ylim(0, 108)
    ax.set_title("Predicted score vs dataset statistics", pad=15, fontweight="bold")
    style_chart(ax)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.2, f"{height:.1f}",
                ha="center", color="#59466C", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    i1, i2, i3 = st.columns(3)
    i1.metric("Difference from average", f"{score - average_score:+.1f} pts")
    i2.metric("Dataset percentile", f"{dataset_percentile:.0f}th")
    i3.metric("Dataset median", f"{median_score:.1f} / 100")

    if score - average_score > .5:
        st.info("The predicted score is above the dataset average. This is a comparison with the dataset, not a guarantee of actual exam performance.")
    elif score - average_score < -.5:
        st.info("The predicted score is below the dataset average. Consider reviewing study habits and seeking academic support where useful.")
    else:
        st.info("The predicted score is close to the dataset average. Actual results may differ from this estimate.")
    st.caption("Percentile compares the predicted score with observed Exam_Score values in this dataset only.")

    # Recommendations
    st.divider()
    section_header("04 · YOUR NEXT STEPS", "Personalized study recommendations",
                   "Rule-based suggestions generated from the information you entered.")
    st.write(f"**{len(recommendations)} recommendation(s) identified.**")
    for index, recommendation in enumerate(recommendations, 1):
        with st.container(border=True):
            st.markdown(
                f'<span class="recommendation-number">{index:02d}</span> **{recommendation["title"]}**',
                unsafe_allow_html=True
            )
            st.write(recommendation["reason"])
            st.markdown(f"**Suggested action**  \n{recommendation['action']}")
    st.caption("These are general suggestions generated using predefined rules, not professional or academic assessments.")

    # What-if simulator
    st.divider()
    section_header("05 · EXPERIMENT", "What-if performance simulator",
                   "Change study hours or attendance and see how the trained model's estimate responds.")
    st.caption("Only these two inputs change. All other student details remain the same.")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        simulated_hours = st.slider("Study hours per week", 1, 44,
                                    int(original_inputs["Hours_Studied"]), key="simulated_hours")
    with sim_col2:
        simulated_attendance = st.slider("Attendance (%)", 60, 100,
                                        int(original_inputs["Attendance"]), key="simulated_attendance")

    simulated_inputs = original_inputs.copy()
    simulated_inputs["Hours_Studied"] = simulated_hours
    simulated_inputs["Attendance"] = simulated_attendance
    simulated_score = predict_score(simulated_inputs)
    difference = simulated_score - score

    r1, r2, r3 = st.columns(3)
    r1.metric("Original estimate", f"{score:.1f} / 100")
    r2.metric("What-if estimate", f"{simulated_score:.1f} / 100")
    r3.metric("Difference", f"{difference:+.1f} pts")

    sim_df = pd.DataFrame({
        "Prediction": ["Original", "What-if simulation"],
        "Score": [score, simulated_score]
    })
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(sim_df["Prediction"], sim_df["Score"],
                  color=["#E8BBD5", "#A78BCB"], width=.48)
    ax.set_ylabel("Predicted exam score")
    ax.set_ylim(0, 108)
    ax.set_title("Original vs simulated prediction", pad=14, fontweight="bold")
    style_chart(ax)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, height+1.2, f"{height:.1f}",
                ha="center", color="#59466C", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Changes made**")
    st.write(f"- Study hours: {original_inputs['Hours_Studied']} → {simulated_hours} hours per week")
    st.write(f"- Attendance: {original_inputs['Attendance']}% → {simulated_attendance}%")
    st.info("This is a model-based what-if estimate, not proof that changing these factors will cause the same improvement in real exam results. The simulation is not saved separately.")

    # PDF report
    st.divider()
    section_header("06 · TAKE IT WITH YOU", "Your individual PDF report",
                   "Download a report with your estimate, submitted details, insights, recommendations and current simulation.")
    pdf_data = create_pdf_report(
        original_inputs, score, average_score, median_score, dataset_percentile,
        recommendations, simulated_hours, simulated_attendance, simulated_score
    )
    st.download_button(
        "↓  Download individual PDF report",
        data=pdf_data,
        file_name="student_performance_report.pdf",
        mime="application/pdf",
        type="primary"
    )
else:
    st.info("Complete the student profile above and select **Generate prediction** to view your results, recommendations, simulator and PDF report.")


# ==================================================
# STUDENT PERFORMANCE DASHBOARD
# ==================================================

st.divider()
section_header("07 · EXPLORE THE DATA", "Student performance dashboard",
               "Understand the score distribution and how selected numerical factors relate to exam scores.")

tab_distribution, tab_correlation = st.tabs(["Score distribution", "Feature correlation"])

with tab_distribution:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(df["Exam_Score"], bins=20, color="#C6A9DD", edgecolor="#FFFFFF", linewidth=1.2)
    ax.set_xlabel("Exam score")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of exam scores", pad=14, fontweight="bold")
    style_chart(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_correlation:
    numeric_columns = [
        "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
        "Tutoring_Sessions", "Physical_Activity", "Exam_Score"
    ]
    correlations = df[numeric_columns].corr()["Exam_Score"].drop("Exam_Score").sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    correlations.plot(kind="barh", ax=ax, color="#B99BCF", edgecolor="white")
    ax.set_xlabel("Correlation with exam score")
    ax.set_ylabel("")
    ax.set_title("Numerical feature correlations", pad=14, fontweight="bold")
    ax.axvline(0, linewidth=1, color="#72538F")
    style_chart(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Correlation describes association in the dataset; it does not establish cause and effect.")


# ==================================================
# MODEL EVALUATION
# ==================================================

st.divider()
section_header("08 · MODEL QUALITY", "Model evaluation",
               "Recorded test-set metrics from the model training stage.")

evaluation = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Decision Tree"],
    "MAE": [0.452, 1.184, 1.600],
    "RMSE": [1.804, 2.233, 2.511],
    "R² Score": [0.770, 0.647, 0.554]
})
st.success("Selected model: Linear Regression. It achieved the best recorded test-set results among the three evaluated models.")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", "0.452")
m2.metric("RMSE", "1.804")
m3.metric("R² score", "0.770")
st.dataframe(evaluation, hide_index=True, use_container_width=True)

with st.expander("Understand the evaluation metrics"):
    st.markdown("""
    **MAE (Mean Absolute Error)**  
    The average absolute difference between predicted and actual scores. Lower is better.

    **RMSE (Root Mean Squared Error)**  
    Measures prediction errors while penalizing larger errors more strongly. Lower is better.

    **R² Score (Coefficient of Determination)**  
    Measures how much variation in the target is explained by the model on the evaluated dataset. Higher is generally better.

    **Important:** R² = 0.770 is not 77% accuracy. These are recorded test-set results and do not guarantee performance for every new student.
    """)


# ==================================================
# PREDICTION HISTORY
# ==================================================

st.divider()
section_header("09 · SAVED ACTIVITY", "Prediction history",
               "Review previous prediction sessions saved in the local SQLite database.")

with sqlite3.connect(DB_PATH) as connection:
    history = pd.read_sql_query(
        "SELECT id, timestamp, predicted_score FROM predictions ORDER BY id DESC LIMIT 10",
        connection
    )
    all_predictions = pd.read_sql_query(
        "SELECT id, timestamp, student_inputs, predicted_score FROM predictions ORDER BY id DESC",
        connection
    )
    summary = pd.read_sql_query(
        """SELECT COUNT(*) AS total_predictions,
                  AVG(predicted_score) AS average_score,
                  MAX(predicted_score) AS highest_score
           FROM predictions""",
        connection
    )

total_predictions = int(summary.loc[0, "total_predictions"])
average_history_score = summary.loc[0, "average_score"]
highest_score = summary.loc[0, "highest_score"]

h1, h2, h3 = st.columns(3)
h1.metric("Total predictions", total_predictions)
h2.metric("Average predicted score", f"{average_history_score:.1f}" if pd.notna(average_history_score) else "N/A")
h3.metric("Highest predicted score", f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A")

if history.empty:
    st.info("No predictions have been recorded yet.")
else:
    st.caption("Showing the 10 most recent predictions.")
    st.dataframe(history, hide_index=True, use_container_width=True)


# ==================================================
# STUDENT PERFORMANCE TREND ANALYSIS
# ==================================================

st.divider()
section_header("10 · YOUR PREDICTION TREND", "Performance trend analysis",
               "Explore how your saved model predictions have varied across sessions.")

if all_predictions.empty:
    st.info("No prediction records are available yet. Make predictions to start building your trend analysis.")
else:
    trend_data = all_predictions.copy()
    trend_data["timestamp"] = pd.to_datetime(trend_data["timestamp"], errors="coerce", utc=True)
    trend_data = trend_data.dropna(subset=["timestamp"]).sort_values("timestamp")

    if trend_data.empty:
        st.info("No valid prediction timestamps are available for trend analysis.")
    else:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total predictions", len(trend_data))
        t2.metric("Average score", f"{trend_data['predicted_score'].mean():.1f}")
        t3.metric("Highest score", f"{trend_data['predicted_score'].max():.1f}")
        t4.metric("Lowest score", f"{trend_data['predicted_score'].min():.1f}")

        trend_data["Prediction Date"] = trend_data["timestamp"].dt.strftime("%d %b %Y, %H:%M")
        chart_data = trend_data[["Prediction Date", "predicted_score"]].set_index("Prediction Date")
        st.line_chart(chart_data, y="predicted_score", color="#9B7ABD", use_container_width=True)
        st.caption("Each point represents a saved model prediction, ordered by prediction time. This is not a record of actual academic progress.")

        with st.expander("View all trend records"):
            display_trend = trend_data[["id", "timestamp", "predicted_score"]].copy()
            display_trend["timestamp"] = display_trend["timestamp"].dt.strftime("%d %b %Y, %H:%M")
            display_trend = display_trend.rename(columns={
                "id": "Prediction ID",
                "timestamp": "Prediction Date",
                "predicted_score": "Predicted Score"
            })
            st.dataframe(display_trend, hide_index=True, use_container_width=True)


# ==================================================
# EXPORT PREDICTION HISTORY
# ==================================================

st.divider()
section_header("11 · EXPORT", "Download your prediction history",
               "Keep a copy of all saved prediction records.")

if all_predictions.empty:
    st.info("Make a prediction before downloading history.")
else:
    csv_data = all_predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓  Download prediction history (CSV)",
        data=csv_data,
        file_name="student_prediction_history.csv",
        mime="text/csv"
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div style="margin-top:2.5rem;padding:1.25rem 1rem;border-top:1px solid #EDE3F2;text-align:center;">
    <div style="font-family:'Playfair Display',serif;color:#654781;font-size:1.05rem;">
        Student Performance Predictor
    </div>
    <div style="color:#8A7B93;font-size:.78rem;margin-top:.35rem;">
        A machine learning-based decision-support and self-assessment tool.
        Predictions are estimates, not official grades or guaranteed results.
    </div>
</div>
""", unsafe_allow_html=True)
