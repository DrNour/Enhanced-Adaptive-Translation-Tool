import streamlit as st
import time
import random
import csv
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
    bert_available = True
except ModuleNotFoundError:
    bert_available = False

# =========================
# Streamlit App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# =========================
# Utility Functions
# =========================
def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

def save_submission(user, source, translation, reference, score_dict):
    with open("submissions.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user, source, translation, reference, str(score_dict), time.time()])

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

def evaluate_translation(student, reference):
    results = {}
    # BLEU/chrF/TER
    if reference and sacrebleu_available:
        results["BLEU"] = sacrebleu.corpus_bleu([student], [[reference]]).score
        results["chrF"] = sacrebleu.corpus_chrf([student], [[reference]]).score
        results["TER"] = sacrebleu.corpus_ter([student], [[reference]]).score
    # Edit distance
    if reference and levenshtein_available:
        results["EditDistance"] = Levenshtein.distance(student, reference)
    # Semantic evaluation
    if bert_available:
        try:
            P, R, F1 = bert_score([student], [reference or student], lang="en", rescale_with_baseline=True)
            results["BERT_F1"] = F1[0].item()
        except Exception as e:
            results["BERT_F1"] = "Error: " + str(e)
    return results

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard/Instructor"])

# =========================
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit")
    source_text = st.text_area("Source Text")
    reference_translation = ""
    if role=="Instructor":
        reference_translation = st.text_area("Reference Translation (Instructor Only)")
    student_translation = st.text_area("Your Translation", height=150)

    start_time = time.time()
    if st.button("Evaluate Translation"):
        score_dict = {}
        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.subheader("💡 Feedback:")
            for f in fb:
                st.warning(f)
            score_dict = evaluate_translation(student_translation, reference_translation)
        else:
            st.info("No reference provided: semantic/fluency evaluation only.")
            score_dict = evaluate_translation(student_translation, None)
        st.write("📊 Evaluation Results", score_dict)
        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} sec")
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")
        save_submission(username, source_text, student_translation, reference_translation, score_dict)

# =========================
# Tab 2: Challenges
# =========================
with tab2:
    st.subheader("⏱️ Timed Challenge")
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
        user_ans = st.text_area("Your Translation (Challenge)", key="challenge_box")
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
# Tab 3: Leaderboard / Instructor
# =========================
with tab3:
    if role=="Student":
        st.subheader("🏆 Leaderboard")
        if st.session_state.leaderboard:
            sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
            for rank, (user, points) in enumerate(sorted_lb, start=1):
                st.write(f"{rank}. **{user}** - {points} points")
        else:
            st.info("No scores yet. Start translating!")
    elif role=="Instructor":
        st.subheader("📊 Instructor Dashboard")
        if pd_available:
            try:
                df = pd.read_csv("submissions.csv", header=None,
                                 names=["Username","Source","Translation","Reference","Scores","Timestamp"])
                st.dataframe(df)
                df_points = df.groupby("Username").size().reset_index(name="Submissions")
                st.bar_chart(df_points.set_index("Username")["Submissions"])
            except FileNotFoundError:
                st.info("No submissions yet.")
        else:
            st.info("Pandas/Matplotlib not installed: dashboard unavailable.")
