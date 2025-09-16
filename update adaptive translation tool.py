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
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_available = True
except ModuleNotFoundError:
    semantic_available = False
    st.warning("sentence-transformers not installed: semantic scoring disabled.")

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
# Idioms/Collocations Pool
# =========================
idioms_en = ["kick the bucket", "break the ice", "once upon a time"]
idioms_ar = ["فارَق الحياة", "كسر الجمود", "كان يا ما كان"]

def check_idioms(student_text, source_lang="en"):
    detected = []
    pool = idioms_en if source_lang=="en" else idioms_ar
    for idiom in pool:
        if idiom.lower() in student_text.lower():
            detected.append(idiom)
    return detected

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
# Semantic Score Function
# =========================
def semantic_score(source_text, student_translation):
    if semantic_available:
        emb_source = st_model.encode(source_text, convert_to_tensor=True)
        emb_student = st_model.encode(student_translation, convert_to_tensor=True)
        score = util.cos_sim(emb_source, emb_student).item()  # 0-1
        return score
    return None

# =========================
# Tabs
# =========================
username = st.text_input("Enter your name:")

tab1, tab2, tab3, tab4 = st.tabs(["Translate & Post-Edit", "Challenges", "Leaderboard", "Instructor Dashboard"])

# =========================
# Tab 1: Translate & Post-Edit
# =========================
with tab1:
    st.subheader("🔎 Translate or Post-Edit")
    source_text = st.text_area("Source Text")
    show_reference = st.checkbox("Show Reference Translation (Instructor Only)", value=False)
    reference_translation = ""
    if show_reference:
        reference_translation = st.text_area("Reference Translation (Human Translation)")

    student_translation = st.text_area("Your Translation", height=150)

    start_time = time.time()
    if st.button("Evaluate Translation"):
        # Highlight literal errors if reference exists
        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            fb = []
            st.info("No reference provided; semantic evaluation only.")

        # Semantic Score
        sem_score = semantic_score(source_text, student_translation)
        if sem_score is not None:
            st.write(f"Semantic Accuracy Score (0-1): {sem_score:.2f}")

        # Idioms / Collocations
        idioms_detected = check_idioms(student_translation)
        if idioms_detected:
            st.success(f"Idioms detected: {', '.join(idioms_detected)}")
        else:
            st.info("No idioms detected. Consider using appropriate idioms/collocations.")

        # BLEU/chrF/TER
        if reference_translation and sacrebleu_available:
            bleu_score = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
            chrf_score = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
            ter_score = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
            st.write(f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, TER: {ter_score:.2f}")

        # Edit Distance
        if reference_translation and levenshtein_available:
            edit_dist = Levenshtein.distance(student_translation, reference_translation)
            st.write(f"Edit Distance: {edit_dist}")

        # Time Taken
        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        # Points
        points = int(sem_score*50) if sem_score else 10
        points += 5*len(idioms_detected)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        # Save feedback
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

            sem_score = semantic_score(st.session_state.challenge[0], user_ans)
            st.write(f"Semantic Accuracy Score (0-1): {sem_score:.2f}" if sem_score else "")

            idioms_detected = check_idioms(user_ans)
            if idioms_detected:
                st.success(f"Idioms detected: {', '.join(idioms_detected)}")
            else:
                st.info("No idioms detected.")

            points = int(sem_score*50) if sem_score else 10
            points += 5*len(idioms_detected)
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
        df = pd.DataFrame([{"Student": user, "Points": points} for user, points in st.session_state.leaderboard.items()])
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

        # Save progress to CSV
        df.to_csv("student_progress.csv", index=False)
        st.success("Progress saved to CSV.")
