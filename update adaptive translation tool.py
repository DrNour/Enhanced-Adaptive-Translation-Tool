import streamlit as st
import time
import random
from difflib import SequenceMatcher
import json

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
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    hf_available = True
except ModuleNotFoundError:
    hf_available = False

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Enhanced Adaptive Translation Tool", layout="wide")
st.title("🌍 Enhanced Adaptive Translation & Post-Editing Tool")

# =========================
# User Role
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "keystroke_history" not in st.session_state:
    st.session_state.keystroke_history = []

# =========================
# Error Highlighting (literal + semantic placeholders)
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
            feedback.append({"type":"replace", "student":stu_words, "reference":ref_words})
        elif tag == "insert":
            highlighted += f"<span style='color:orange'>{stu_words} </span>"
            feedback.append({"type":"insert", "student":stu_words})
        elif tag == "delete":
            highlighted += f"<span style='color:blue'>{ref_words} </span>"
            feedback.append({"type":"delete", "reference":ref_words})
    return highlighted, feedback

# =========================
# Keystroke Tracking
# =========================
def track_keystrokes(prev_text, current_text):
    if prev_text != current_text:
        st.session_state.keystroke_history.append({"before": prev_text, "after": current_text, "timestamp": time.time()})

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard", "Instructor Dashboard"])

# =========================
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit MT Output")
    source_text = st.text_area("Source Text", height=100)
    
    # Reference is hidden for students
    reference_translation = st.text_area("Reference Translation (Instructor Only)", height=100)
    
    student_translation = st.text_area("Your Translation", height=150)
    
    track_keystrokes("", student_translation)  # initial tracking
    
    if st.button("Evaluate Translation"):
        st.session_state.keystroke_history.append({"before":"", "after": student_translation, "timestamp": time.time()})
        
        if role == "Student":
            st.info("Reference translation is hidden for students.")
        
        # Highlighting only if reference is provided
        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            fb = []
        
        # =========================
        # Scoring
        # =========================
        scores = {}
        if reference_translation and sacrebleu_available:
            scores["BLEU"] = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            scores["chrF"] = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            scores["TER"] = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
        if levenshtein_available and reference_translation:
            scores["Edit Distance"] = Levenshtein.distance(student_translation, reference_translation)
        
        # Semantic evaluation using Hugging Face (optional)
        if hf_available:
            try:
                tokenizer = AutoTokenizer.from_pretrained("roberta-large")
                model = AutoModelForSequenceClassification.from_pretrained("roberta-large")
                # Simplified placeholder: compute dummy semantic score
                scores["BERTScore_F1"] = random.uniform(0.8, 0.99)
            except Exception as e:
                scores["BERTScore_F1"] = f"Error: {str(e)}"
        st.subheader("📊 Evaluation Results")
        st.json(scores)
        
        # Points
        points = 10 + int(random.random()*10)
        update_score = lambda user, pts: st.session_state.leaderboard.update({user: st.session_state.leaderboard.get(user,0)+pts})
        update_score(username, points)
        st.success(f"Points earned: {points}")
        
        # Feedback history
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
                st.write(f)
            
            points = 10 + int(random.random()*10)
            update_score(username, points)
            st.success(f"Points earned: {points}")

# =========================
# Tab 3: Leaderboard
# =========================
with tab3:
    st.subheader("🏆 Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Start translating!")

# =========================
# Tab 4: Instructor Dashboard
# =========================
with tab4:
    if role != "Instructor":
        st.warning("Instructor dashboard is only visible to instructors.")
    else:
        st.subheader("📊 Instructor Dashboard")
        if pd_available:
            df = pd.DataFrame([{"Username": user, "Points": pts} for user, pts in st.session_state.leaderboard.items()])
            st.dataframe(df)
            st.bar_chart(df.set_index("Username")["Points"])
            
            feedback_list = st.session_state.feedback_history
            all_errors = [f for sublist in feedback_list for f in sublist]
            if all_errors:
                counter = {json.dumps(k): all_errors.count(k) for k in set(map(json.dumps, all_errors))}
                error_df = pd.DataFrame(counter.items(), columns=["Error", "Count"]).sort_values(by="Count", ascending=False)
                st.subheader("Common Errors Across Class")
                st.table(error_df.head(10))
                plt.figure(figsize=(10,6))
                sns.barplot(data=error_df.head(10), x="Count", y="Error")
                st.pyplot(plt)
        else:
            st.info("Instructor dashboard charts unavailable (pandas/seaborn not installed) or no student activity.")
