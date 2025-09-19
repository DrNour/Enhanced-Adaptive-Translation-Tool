import streamlit as st
import time
import random
from difflib import SequenceMatcher

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False

try:
    import Levenshtein
    levenshtein_available = True
except ModuleNotFoundError:
    levenshtein_available = False

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False

try:
    from bert_score import score as bert_score
    bert_score_available = True
except ModuleNotFoundError:
    bert_score_available = False

try:
    from comet_ml import OfflineExperiment
    comet_available = True
except ModuleNotFoundError:
    comet_available = False

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# User Role
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# Initialize session state
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

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

# =========================
# Student Interface
# =========================
if role == "Student":
    st.subheader("🔎 Translate or Post-Edit MT Output")
    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (optional)")
    student_translation = st.text_area("Your Translation", height=150)

    start_time = time.time()
    if st.button("Evaluate Translation"):
        if reference_translation.strip():
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            fb = []

        # Scores
        if reference_translation.strip() and sacrebleu_available:
            bleu_score = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter_score = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}")
        else:
            st.info("BLEU/chrF/TER disabled (no reference or sacrebleu missing).")

        if levenshtein_available and reference_translation.strip():
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"Edit Distance: {edit_dist}")
        else:
            st.info("Edit Distance disabled (no reference or Levenshtein missing).")

        # Semantic evaluation
        if bert_score_available:
            try:
                P, R, F1 = bert_score([student_translation], [reference_translation if reference_translation.strip() else student_translation], lang="en")
                st.write(f"BERTScore F1: {F1[0]:.4f}")
            except Exception as e:
                st.warning(f"Semantic evaluation error: {e}")
        else:
            st.info("Semantic evaluation disabled (bert_score not installed).")

        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Points
        points = 10 + int(random.random() * 10)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        st.session_state.feedback_history.append(fb)

# =========================
# Instructor Interface
# =========================
if role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    if pd_available and st.session_state.leaderboard:
        df = pd.DataFrame([{"Username": user, "Points": pts} for user, pts in st.session_state.leaderboard.items()])
        st.dataframe(df)
        st.bar_chart(df.set_index("Username")["Points"])

        feedback_list = st.session_state.feedback_history
        all_errors = [f for sublist in feedback_list for f in sublist]
        if all_errors:
            counter = {k: all_errors.count(k) for k in set(all_errors)}
            error_df = pd.DataFrame(counter.items(), columns=["Error", "Count"]).sort_values(by="Count", ascending=False)
            st.subheader("Common Errors Across Class")
            st.table(error_df.head(10))
            plt.figure(figsize=(10,6))
            sns.barplot(data=error_df.head(10), x="Count", y="Error")
            st.pyplot(plt)
    else:
        st.info("No student activity yet or pandas/seaborn not installed.")
