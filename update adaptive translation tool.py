import streamlit as st
from difflib import SequenceMatcher
import time
import random
import csv

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER scoring will be disabled.")

try:
    import Levenshtein
    levenshtein_available = True
except ModuleNotFoundError:
    levenshtein_available = False
    st.warning("python-Levenshtein not installed: edit distance scoring will be disabled.")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False
    st.warning("pandas/seaborn/matplotlib not installed: Instructor dashboard charts disabled.")

try:
    from sentence_transformers import SentenceTransformer, util
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_available = True
except ModuleNotFoundError:
    semantic_available = False
    st.warning("sentence-transformers not installed: reference-free semantic evaluation disabled.")

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# Gamification / Leaderboard
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# Idioms & Collocations (Instructor-provided)
# =========================
idioms_dict = {
    "Literary": ["break the ice", "once upon a time"],
    "Legal": ["beyond reasonable doubt", "due diligence"],
    "Journalistic": ["breaking news", "on the record"],
    "Scientific": ["statistical significance", "control group"]
}

def check_idioms(text, genre):
    feedback = []
    for idiom in idioms_dict.get(genre, []):
        if idiom not in text:
            feedback.append(f"Consider using the idiom/collocation: '{idiom}'")
    return feedback

# =========================
# Error Highlighting Function
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
# Semantic Evaluation
# =========================
def semantic_score(source, student):
    if semantic_available:
        source_emb = semantic_model.encode(source, convert_to_tensor=True)
        student_emb = semantic_model.encode(student, convert_to_tensor=True)
        score = util.pytorch_cos_sim(source_emb, student_emb).item()
        return round(score, 2)
    else:
        return None

# =========================
# CSV Progress Tracking
# =========================
def save_progress(username, genre, source, student, points, semantic=None):
    with open("progress.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), username, genre, source, student, points, semantic])

# =========================
# Tabs
# =========================
username = st.text_input("Enter your name:")

tab1, tab2, tab3, tab4 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard", "Instructor Dashboard"])

# =========================
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit MT Output")
    genre = st.selectbox("Select Text Genre", ["Literary", "Legal", "Journalistic", "Scientific"])
    source_text = st.text_area("Source Text")
    reference_translation = st.text_area("Reference Translation (Instructor Only)")
    student_translation = st.text_area("Your Translation", height=150)
    reference_free = st.checkbox("Reference-Free Evaluation (no human translation)")

    start_time = time.time()
    if st.button("Evaluate Translation"):
        feedback = []
        if not reference_free and reference_translation.strip():
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
            feedback.extend(fb)
        else:
            st.info("Reference-free mode: direct semantic assessment.")

        # Idioms/collocations feedback
        idiom_feedback = check_idioms(student_translation, genre)
        feedback.extend(idiom_feedback)

        # Display feedback
        st.subheader("💡 Feedback:")
        if feedback:
            for f in feedback:
                st.warning(f)
        else:
            st.success("No errors detected. Good job!")

        # Scores
        if not reference_free and sacrebleu_available and reference_translation.strip():
            bleu_score = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter_score = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}")
        if levenshtein_available and not reference_free and reference_translation.strip():
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"Edit Distance: {edit_dist}")

        # Semantic score
        sem_score = semantic_score(source_text, student_translation)
        if sem_score is not None:
            st.write(f"Semantic Adequacy Score (0-1): {sem_score}")

        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Points
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        # Save feedback history & progress
        st.session_state.feedback_history.append(feedback)
        save_progress(username, genre, source_text, student_translation, points, sem_score)

# =========================
# Tab 2: Challenges
# =========================
with tab2:
    st.subheader("⏱️ Timer Challenge Mode")
    challenges = [
        ("I love you.", "أنا أحبك.", "Literary"),
        ("Knowledge is power.", "المعرفة قوة.", "Scientific"),
        ("The weather is nice today.", "الطقس جميل اليوم.", "Journalistic")
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
            idiom_feedback = check_idioms(user_ans, st.session_state.challenge[2])
            fb.extend(idiom_feedback)
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
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {points} points")
    else:
        st.info("No scores yet. Start translating!")

# =========================
# Tab 4: Instructor Dashboard
# =========================
with tab4:
    st.subheader("📊 Instructor Dashboard")
    if pd_available and st.session_state.leaderboard:
        df = pd.DataFrame([
            {"Student": user, "Points": points} 
            for user, points in st.session_state.leaderboard.items()
        ])
        st.dataframe(df)
        st.bar_chart(df.set_index("Student")["Points"])
        
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
        st.info("Instructor dashboard charts unavailable (pandas/seaborn not installed) or no student activity.")
