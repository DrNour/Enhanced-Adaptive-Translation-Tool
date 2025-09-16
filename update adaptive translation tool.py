import streamlit as st
from difflib import SequenceMatcher
import time
import random
import csv
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
    from sentence_transformers import SentenceTransformer, util
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_available = True
except Exception:
    semantic_available = False
    st.warning("sentence-transformers not installed: Semantic scoring disabled.")

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# Role Selection
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])

# =========================
# Session State Initialization
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "exercise_history" not in st.session_state:
    st.session_state.exercise_history = []

username = st.text_input("Enter your name:")

# =========================
# Helper Functions
# =========================
def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

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

def semantic_score(source, translation):
    if not semantic_available:
        return None
    emb1 = st_model.encode(source, convert_to_tensor=True)
    emb2 = st_model.encode(translation, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score

def save_progress_csv(username, role, source, student_trans, points):
    file_exists = os.path.isfile("progress.csv")
    with open("progress.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Username", "Role", "Source", "Translation", "Points", "Timestamp"])
        writer.writerow([username, role, source, student_trans, points, time.time()])

# =========================
# Student Interface
# =========================
if role == "Student":
    st.subheader("🔎 Translate / Post-Edit")
    source_text = st.text_area("Source Text")
    student_translation = st.text_area("Your Translation", height=150)
    reference_translation = st.text_area("Reference Translation (Optional, Instructor Only)", height=150)

    start_time = time.time()
    if st.button("Evaluate Translation"):
        # Feedback
        fb = []
        highlighted = ""
        if reference_translation.strip():
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)

        # Scores
        if reference_translation.strip() and sacrebleu_available:
            bleu = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            st.write(f"BLEU: {bleu:.2f}, chrF: {chrf:.2f}, TER: {ter:.2f}")
        if levenshtein_available and reference_translation.strip():
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"Edit Distance: {edit_dist}")
        # Semantic
        if semantic_available:
            sem_score = semantic_score(source_text, student_translation)
            if sem_score is not None:
                st.write(f"Semantic Adequacy (0-1): {sem_score:.2f}")

        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Points & Feedback history
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")
        st.session_state.feedback_history.append(fb)
        save_progress_csv(username, "Student", source_text, student_translation, points)

# =========================
# Instructor Interface
# =========================
elif role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    st.write("View all student activity and manage exercises.")

    if pd_available:
        if os.path.exists("progress.csv"):
            df = pd.read_csv("progress.csv")
            st.dataframe(df)

            # Leaderboard
            leaderboard_df = df.groupby("Username")["Points"].sum().reset_index().sort_values(by="Points", ascending=False)
            st.subheader("🏆 Leaderboard")
            st.table(leaderboard_df)

            # Common errors (from feedback_history)
            feedback_list = st.session_state.feedback_history
            all_errors = [f for sublist in feedback_list for f in sublist]
            if all_errors:
                counter = {k: all_errors.count(k) for k in set(all_errors)}
                error_df = pd.DataFrame(counter.items(), columns=["Error", "Count"]).sort_values(by="Count", ascending=False)
                st.subheader("Common Errors")
                st.table(error_df.head(10))
                plt.figure(figsize=(10,6))
                sns.barplot(data=error_df.head(10), x="Count", y="Error")
                st.pyplot(plt)

    # Exercise Management
    st.subheader("✏️ Create / Manage Exercises")
    new_source = st.text_area("New Source Text")
    new_reference = st.text_area("Reference Translation (Optional)")
    new_idiom = st.text_input("Suggested Idiom / Collocation")
    if st.button("Add Exercise"):
        st.session_state.exercise_history.append({
            "source": new_source,
            "reference": new_reference,
            "idiom": new_idiom
        })
        st.success("Exercise added!")

    # View Exercises
    if st.session_state.exercise_history:
        st.subheader("📚 All Exercises")
        for ex in st.session_state.exercise_history:
            st.write(f"Source: {ex['source']}")
            st.write(f"Reference: {ex['reference']}")
            st.write(f"Suggested Idiom/Collocation: {ex['idiom']}")
