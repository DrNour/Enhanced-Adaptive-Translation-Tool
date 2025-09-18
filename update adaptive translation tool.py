# adaptive_translation_prize.py
import streamlit as st
from difflib import SequenceMatcher
import time, random, csv
import os

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER scoring disabled.")

try:
    import Levenshtein
    levenshtein_available = True
except ModuleNotFoundError:
    levenshtein_available = False
    st.warning("python-Levenshtein not installed: Edit distance disabled.")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False
    st.warning("pandas/seaborn/matplotlib not installed: Dashboard charts disabled.")

try:
    from bert_score import score as bert_score
    bertscore_available = True
except ModuleNotFoundError:
    bertscore_available = False
    st.warning("bert-score not installed: Semantic evaluation disabled.")

# =========================
# CSV Tracking
# =========================
CSV_FILE = "student_activity.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "SourceText", "StudentTranslation", "ReferenceTranslation", "Points", "TimeTaken"])

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# User role selection
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# Session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# =========================
# Gamification Functions
# =========================
def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

def save_to_csv(username, source, student, reference, points, time_taken):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, source, student, reference, points, round(time_taken,2)])

# =========================
# Highlighting Edits
# =========================
def highlight_diff(student, reference):
    matcher = SequenceMatcher(None, reference.split(), student.split())
    highlighted = ""
    feedback = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        stu_words = " ".join(student.split()[j1:j2])
        ref_words = " ".join(reference.split()[i1:i2])
        if tag == "equal":
            highlighted += f"<span style='color:green'>{stu_words} </span>"
        elif tag == "replace":
            highlighted += f"<span style='color:red'>{stu_words} </span>"
            feedback.append(f"Replace '{stu_words}' with '{ref_words}'")
        elif tag == "insert":
            highlighted += f"<span style='color:orange'>{stu_words} </span>"
            feedback.append(f"Extra words: '{stu_words}'")
        elif tag == "delete":
            highlighted += f"<span style='color:blue'>{ref_words} </span>"
            feedback.append(f"Missing: '{ref_words}'")
    return highlighted, feedback

# =========================
# Evaluation
# =========================
def evaluate_translation(student, reference=None):
    results = {}
    if reference:
        if sacrebleu_available:
            try:
                results["BLEU"] = sacrebleu.corpus_bleu([student],[ [reference] ]).score
                results["chrF"] = sacrebleu.corpus_chrf([student],[ [reference] ]).score
                results["TER"] = sacrebleu.corpus_ter([student],[ [reference] ]).score
            except Exception as e:
                results["Error_metrics"] = str(e)
        if levenshtein_available:
            try:
                results["Edit Distance"] = Levenshtein.distance(student, reference)
            except Exception as e:
                results["Error_edit"] = str(e)
        if bertscore_available:
            try:
                P, R, F1 = bert_score([student], [reference], lang="en", rescale_with_baseline=True)
                results["BERTScore_F1"] = round(F1.mean().item(),3)
            except Exception as e:
                results["Error_bert"] = str(e)
    else:
        results["Info"] = "Reference translation not provided; only basic feedback."
    return results

# =========================
# Student Interface
# =========================
if role == "Student":
    st.subheader("🔎 Translate or Post-Edit")
    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (optional)")
    student_translation = st.text_area("Your Translation", height=150)
    
    start_time = time.time()
    if st.button("Evaluate Translation"):
        elapsed_time = time.time() - start_time
        if reference_translation.strip():
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)
        results = evaluate_translation(student_translation, reference_translation if reference_translation.strip() else None)
        st.subheader("📊 Evaluation Results")
        for metric, value in results.items():
            st.write(f"**{metric}:** {value}")
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")
        save_to_csv(username, source_text, student_translation, reference_translation, points, elapsed_time)
        st.session_state.feedback_history.append(student_translation)

# =========================
# Instructor Interface
# =========================
elif role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    if pd_available and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            st.dataframe(df)
            st.bar_chart(df.groupby("Username")["Points"].sum())
            # Common submissions
            sub_list = df["StudentTranslation"].tolist()
            counter = {f: sub_list.count(f) for f in set(sub_list)}
            error_df = pd.DataFrame(counter.items(), columns=["Submission", "Count"]).sort_values(by="Count", ascending=False)
            st.subheader("Student Submissions Summary")
            st.table(error_df.head(10))
            plt.figure(figsize=(10,6))
            sns.barplot(data=error_df.head(10), x="Count", y="Submission")
            st.pyplot(plt)
        else:
            st.info("No student submissions yet.")
    else:
        st.info("Dashboard unavailable: pandas not installed or CSV missing.")
