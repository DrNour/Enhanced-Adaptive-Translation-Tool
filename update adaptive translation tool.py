import streamlit as st
import time
import random
from difflib import SequenceMatcher
import os
import csv

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
    bert_available = True
except ModuleNotFoundError:
    bert_available = False
    st.warning("bert_score not installed: Semantic evaluation disabled.")

# =========================
# Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# User Role
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# CSV file for tracking
TRACK_FILE = "student_progress.csv"
if not os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "Points", "Time", "Source", "Submission"])

# =========================
# Functions
# =========================
def update_csv(username, points, elapsed_time, source, submission):
    with open(TRACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, points, round(elapsed_time, 2), source, submission])

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

def calculate_scores(student, reference):
    scores = {}
    if sacrebleu_available and reference:
        scores["BLEU"] = sacrebleu.corpus_bleu([student], [[reference]]).score
        scores["chrF"] = sacrebleu.corpus_chrf([student], [[reference]]).score
        scores["TER"] = sacrebleu.corpus_ter([student], [[reference]]).score
    if levenshtein_available:
        scores["Edit Distance"] = Levenshtein.distance(student, reference) if reference else "N/A"
    if bert_available:
        if reference:
            P, R, F1 = bert_score([student], [reference], lang="en")
            scores["BERTScore"] = float(F1.mean())
        else:
            scores["BERTScore"] = "N/A"
    return scores

def suggest_idioms(text):
    idioms = ["break the ice", "once upon a time", "kick the bucket"]
    suggestions = [i for i in idioms if i not in text.lower()]
    return suggestions

# =========================
# Student View
# =========================
if role == "Student":
    st.subheader("🔎 Translate / Post-Edit")

    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (optional)")
    student_translation = st.text_area("Your Translation", height=150)

    start_time = time.time()
    if st.button("Submit Translation"):
        elapsed_time = time.time() - start_time

        # Highlight diff if reference is provided
        if reference_translation.strip():
            highlighted, feedback = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in feedback:
                st.warning(f)
        else:
            st.info("Reference not provided: only semantic evaluation applied.")

        # Calculate scores
        scores = calculate_scores(student_translation, reference_translation)
        st.subheader("📊 Scores:")
        for k, v in scores.items():
            st.write(f"{k}: {v}")

        # Idiom suggestions
        idiom_suggestions = suggest_idioms(student_translation)
        if idiom_suggestions:
            st.subheader("💬 Idiom/Collocation Suggestions:")
            for i in idiom_suggestions:
                st.info(f"Consider using: '{i}'")

        # Points & CSV tracking
        points = random.randint(5, 15)
        update_csv(username, points, elapsed_time, source_text, student_translation)
        st.success(f"Points earned: {points}")
        st.write(f"Time taken: {round(elapsed_time,2)} seconds")

# =========================
# Instructor View
# =========================
if role == "Instructor":
    st.subheader("📊 Instructor Dashboard")

    if os.path.exists(TRACK_FILE):
        df = pd.read_csv(TRACK_FILE) if pd_available else None
        if pd_available and not df.empty:
            leaderboard_df = df.groupby("Username")["Points"].sum().reset_index().sort_values(by="Points", ascending=False)
            st.write("🏆 Leaderboard")
            st.dataframe(leaderboard_df)

            # Bar chart
            st.subheader("📈 Points Chart")
            st.bar_chart(leaderboard_df.set_index("Username")["Points"])

            # Common errors
            st.subheader("📌 Submissions")
            st.dataframe(df)
        else:
            st.info("No data yet or pandas not available.")
    else:
        st.info("No submissions yet.")
