# enhanced_translation_tool.py
import streamlit as st
import time
import random
import json
from difflib import SequenceMatcher
import sqlite3

# =========================
# Optional packages
# =========================
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
    from bert_score import score as bert_score
    bert_available = True
except ModuleNotFoundError:
    bert_available = False

try:
    from comet_ml import download_model, load_from_checkpoint
    comet_available = True
except ModuleNotFoundError:
    comet_available = False

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# SQLite backend
# =========================
conn = sqlite3.connect('translations.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS submissions 
             (username TEXT, source TEXT, student_translation TEXT, timestamp REAL, score REAL)''')
conn.commit()

# =========================
# Session State
# =========================
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# =========================
# Utility Functions
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

def compute_metrics(student, reference):
    results = {}
    if sacrebleu_available and reference:
        results["BLEU"] = sacrebleu.corpus_bleu([student], [[reference]]).score
        results["chrF"] = sacrebleu.corpus_chrf([student], [[reference]]).score
        results["TER"] = sacrebleu.corpus_ter([student], [[reference]]).score
    if levenshtein_available:
        results["Edit Distance"] = Levenshtein.distance(student, reference if reference else "")
    if bert_available:
        try:
            P, R, F1 = bert_score([student], [reference] if reference else [student], lang='en')
            results["BERT_F1"] = float(F1[0])
        except Exception as e:
            results["BERT_F1"] = f"Error: {e}"
    return results

# =========================
# Sidebar: User type & name
# =========================
user_type = st.sidebar.radio("I am a:", ["Student", "Instructor"])
username = st.sidebar.text_input("Enter your name:")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Translate / Submit", "Leaderboard", "Instructor Dashboard"])

# =========================
# Tab 1: Translate / Submit
# =========================
with tab1:
    st.subheader("🔎 Translate / Submit Your Work")
    source_text = st.text_area("Source Text")
    
    # Instructor only: reference translation
    reference_translation = ""
    if user_type == "Instructor":
        reference_translation = st.text_area("Reference Translation (Instructor Only)")
    
    student_translation = st.text_area("Your Translation", height=150)

    if st.button("Submit Translation"):
        start_time = time.time()
        
        # Metrics computation
        results = compute_metrics(student_translation, reference_translation if reference_translation else None)
        elapsed_time = time.time() - start_time
        
        # Display feedback
        st.subheader("📊 Evaluation Results")
        st.json(results)
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")
        
        # Update leaderboard
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")
        
        # Save to SQLite
        c.execute("INSERT INTO submissions VALUES (?,?,?, ?, ?)", 
                  (username, source_text, student_translation, time.time(), points))
        conn.commit()
        
        # Highlight diff if reference available
        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)
            st.session_state.feedback_history.append(fb)
        
        # Idioms/Collocations suggestions (optional)
        st.subheader("💬 Idioms / Collocations Suggestions")
        idioms_list = ["break the ice", "once upon a time"]
        for idiom in idioms_list:
            st.info(f"Consider using: '{idiom}'")

# =========================
# Tab 2: Leaderboard
# =========================
with tab2:
    st.subheader("🏆 Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Submit translations!")

# =========================
# Tab 3: Instructor Dashboard
# =========================
with tab3:
    if user_type != "Instructor":
        st.info("Instructor dashboard visible only to instructors.")
    else:
        st.subheader("📊 Instructor Dashboard")
        df = []
        for row in c.execute("SELECT * FROM submissions"):
            df.append({
                "Username": row[0], 
                "Source": row[1], 
                "Translation": row[2],
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row[3])),
                "Score": row[4]
            })
        if df:
            st.dataframe(df)
        else:
            st.info("No submissions yet.")
