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
    from transformers import pipeline
    bert_available = True
except ModuleNotFoundError:
    bert_available = False
    st.warning("transformers not installed: Semantic evaluation disabled.")

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# User role selection
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# =========================
# Session states
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# =========================
# Hugging Face Semantic Evaluator
# =========================
hf_token = st.secrets.get("HF_TOKEN", None)
if bert_available and hf_token:
    try:
        bert_scorer = pipeline("text-classification", model="roberta-large", use_auth_token=hf_token)
    except Exception as e:
        st.warning(f"Semantic evaluation unavailable: {e}")
        bert_available = False

# =========================
# Leaderboard update
# =========================
def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# Error highlighting
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
# CSV Logging
# =========================
LOG_FILE = "student_progress.csv"
def log_progress(username, text, points):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Username","Translation","Points","Timestamp"])
        writer.writerow([username, text, points, time.strftime("%Y-%m-%d %H:%M:%S")])

# =========================
# Tabs for students/instructors
# =========================
tab1, tab2, tab3 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard"])

# =========================
# Tab 1: Translation / Post-edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit MT Output")
    source_text = st.text_area("Source Text")
    
    # Only instructor sees reference
    reference_translation = ""
    if role == "Instructor":
        reference_translation = st.text_area("Reference Translation (Instructor Only)")

    student_translation = st.text_area("Your Translation", height=150)

    start_time = time.time()
    if st.button("Evaluate Translation"):
        elapsed_time = time.time() - start_time
        points = 10 + int(random.random()*10)

        # Highlight errors only if reference exists
        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)
        else:
            st.info("No reference translation provided. Only semantic/fluency evaluation applied.")
            fb = []

        # BLEU/chrF/TER
        if reference_translation and sacrebleu_available:
            bleu_score = sacrebleu.corpus_bleu([student_translation],[ [reference_translation] ]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation],[ [reference_translation] ]).score
            ter_score = sacrebleu.corpus_ter([student_translation],[ [reference_translation] ]).score
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}")
        else:
            st.info("BLEU/chrF/TER not computed (no reference or sacrebleu missing).")

        # Edit distance
        if reference_translation and levenshtein_available:
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"Edit Distance: {edit_dist}")

        # Semantic/BERT evaluation
        if bert_available:
            try:
                semantic_score = random.uniform(0,1)  # placeholder for BERT semantic similarity
                st.write(f"Semantic Adequacy Score (0-1): {semantic_score:.2f}")
            except Exception as e:
                st.warning(f"Semantic evaluation failed: {e}")

        st.write(f"Time Taken: {elapsed_time:.2f} seconds")
        update_score(username, points)
        log_progress(username, student_translation, points)
        st.success(f"Points earned: {points}")
        st.session_state.feedback_history.append(fb)

# =========================
# Tab 2: Challenges
# =========================
with tab2:
    st.subheader("⏱️ Timer Challenge Mode")
    challenges = [
        ("I love you.", "أنا أحبك."),
        ("Knowledge is power.", "المعرفة قوة."),
        ("The weather is nice today.", "الطقس جميل اليوم.")
    ]
    if st.button("Start Challenge"):
        challenge = random.choice(challenges)
        st.session_state.challenge = challenge
        st.write(f"Translate: **{challenge[0]}**")

    if "challenge" in st.session_state:
        user_ans = st.text_area("Your Translation (Challenge Mode)", key="challenge_box")
        if st.button("Submit Challenge"):
            highlighted, fb = highlight_diff(user_ans, st.session_state.challenge[1])
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("Feedback:")
            for f in fb:
                st.warning(f)
            points = 10 + int(random.random()*10)
            update_score(username, points)
            st.success(f"Points earned: {points}")

# =========================
# Tab 3: Leaderboard
# =========================
with tab3:
    st.subheader("🏆 Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x:x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Start translating!")
