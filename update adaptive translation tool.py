import streamlit as st
import time
import random
from difflib import SequenceMatcher
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
    from bert_score import score as bert_score
    bert_available = True
except ModuleNotFoundError:
    bert_available = False
    st.warning("bert_score not installed: Semantic evaluation disabled.")

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# CSV storage
DATA_FILE = "student_submissions.csv"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp","Username","Source","StudentTranslation","ReferenceTranslation","IdiomsCollocations"])

# =========================
# Session State
# =========================
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# =========================
# Functions
# =========================
def update_score(username, points):
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

def save_submission(username, source, student, reference, idioms):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, username, source, student, reference, idioms])

# =========================
# User Role
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# =========================
# Tabs
# =========================
if role=="Student":
    tab1, tab2 = st.tabs(["Translate & Post-Edit", "Challenge"])
    
    with tab1:
        st.subheader("Translate or Post-Edit")
        source_text = st.text_area("Source Text")
        student_translation = st.text_area("Your Translation", height=150)
        idioms = st.text_input("Idioms/Collocations (optional)")
        
        if st.button("Submit Translation"):
            # Feedback
            reference_translation = ""  # Student never sees reference
            highlighted, fb = "", []
            if reference_translation:
                highlighted, fb = highlight_diff(student_translation, reference_translation)
            
            st.subheader("Feedback:")
            for f in fb:
                st.warning(f)
            
            # Scores
            if reference_translation and sacrebleu_available:
                bleu = sacrebleu.corpus_bleu([student_translation],[ [reference_translation] ]).score
                chrf = sacrebleu.corpus_chrf([student_translation],[ [reference_translation] ]).score
                ter = sacrebleu.corpus_ter([student_translation],[ [reference_translation] ]).score
                st.write(f"BLEU: {bleu:.2f}, chrF: {chrf:.2f}, TER: {ter:.2f}")
            
            if levenshtein_available:
                edit_dist = Levenshtein.distance(student_translation, reference_translation or "")
                st.write(f"Edit Distance: {edit_dist}")
            
            if bert_available:
                try:
                    P,R,F1 = bert_score([student_translation],[reference_translation or student_translation], lang="en")
                    st.write(f"BERTScore F1: {F1[0].item():.4f}")
                except:
                    st.write("BERTScore unavailable.")
            
            points = random.randint(10,20)
            update_score(username, points)
            st.success(f"Points earned: {points}")
            
            save_submission(username, source_text, student_translation, reference_translation, idioms)
    
    with tab2:
        st.subheader("Challenges")
        st.info("Instructor-provided challenges will appear here (not implemented yet).")
    
elif role=="Instructor":
    tab1, tab2, tab3 = st.tabs(["Dashboard","Add Exercises","Leaderboard"])
    
    with tab1:
        st.subheader("Instructor Dashboard")
        if pd_available and os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.dataframe(df)
            
            # Points leaderboard
            leaderboard_df = df.groupby("Username")["StudentTranslation"].count().reset_index().sort_values(by="StudentTranslation", ascending=False)
            st.subheader("Submission Count Leaderboard")
            st.dataframe(leaderboard_df)
            
            # Error frequency
            all_feedback = []
            for stxt, ref in zip(df["StudentTranslation"], df["ReferenceTranslation"]):
                if ref:
                    _, fb = highlight_diff(stxt, ref)
                    all_feedback.extend(fb)
            if all_feedback:
                counter = {k: all_feedback.count(k) for k in set(all_feedback)}
                error_df = pd.DataFrame(counter.items(), columns=["Error","Count"]).sort_values(by="Count", ascending=False)
                st.subheader("Common Errors")
                st.table(error_df.head(10))
                plt.figure(figsize=(10,6))
                sns.barplot(data=error_df.head(10), y="Error", x="Count")
                st.pyplot(plt)
    
    with tab2:
        st.subheader("Add Exercises")
        source_ex = st.text_area("Source Text for Exercise")
        ref_ex = st.text_area("Reference Translation")
        idioms_ex = st.text_input("Idioms/Collocations")
        if st.button("Add Exercise"):
            save_submission("Instructor", source_ex, "", ref_ex, idioms_ex)
            st.success("Exercise saved.")
    
    with tab3:
        st.subheader("Leaderboard")
        if pd_available and os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            lb = df.groupby("Username")["StudentTranslation"].count().reset_index().sort_values(by="StudentTranslation", ascending=False)
            st.dataframe(lb)
