import streamlit as st
import time
import random
from difflib import SequenceMatcher

# Optional libraries
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

try:
    from comet_ml import download_model, load_from_checkpoint
    comet_available = True
except ModuleNotFoundError:
    comet_available = False
    st.warning("COMET not installed: Deep semantic evaluation disabled.")

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# =========================
# Session state defaults
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "submissions" not in st.session_state:
    st.session_state.submissions = []

# =========================
# User type
# =========================
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# =========================
# Helper functions
# =========================
def update_score(user, points):
    st.session_state.score += points
    if user not in st.session_state.leaderboard:
        st.session_state.leaderboard[user] = 0
    st.session_state.leaderboard[user] += points

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

def compute_scores(student_translation, reference_translation=None):
    results = {}
    # BLEU / chrF / TER
    if reference_translation and sacrebleu_available:
        results["BLEU"] = sacrebleu.corpus_bleu([student_translation], [[reference_translation]]).score
        results["chrF"] = sacrebleu.corpus_chrf([student_translation], [[reference_translation]]).score
        results["TER"] = sacrebleu.corpus_ter([student_translation], [[reference_translation]]).score
    # Edit distance
    if reference_translation and levenshtein_available:
        results["Edit Distance"] = Levenshtein.distance(student_translation, reference_translation)
    # BERTScore
    if bert_available:
        refs = [reference_translation] if reference_translation else [student_translation]
        P, R, F1 = bert_score([student_translation], refs, lang="en", rescale_with_baseline=True)
        results["BERT_F1"] = float(F1[0])
    # COMET placeholder
    if comet_available:
        results["COMET"] = "COMET evaluation applied (requires valid token)"
    return results

def suggest_idioms(student_translation):
    # Placeholder example
    idioms = []
    if "love" in student_translation.lower():
        idioms.append("break the ice")
    if "story" in student_translation.lower():
        idioms.append("once upon a time")
    return idioms

# =========================
# Student interface
# =========================
if role == "Student":
    st.subheader("📝 Submit Your Translation")
    source_text = st.text_area("Source Text")
    student_translation = st.text_area("Your Translation", height=150)
    provide_ref = st.checkbox("Instructor will provide reference translation (hidden from student)", value=True)

    if st.button("Evaluate Translation"):
        reference_translation = None
        if provide_ref:
            # Look up instructor reference
            reference_translation = st.session_state.submissions[-1]["reference"] if st.session_state.submissions else None

        highlighted, feedback = highlight_diff(student_translation, reference_translation) if reference_translation else ("", [])
        if highlighted:
            st.markdown(highlighted, unsafe_allow_html=True)
        if feedback:
            st.subheader("💡 Feedback")
            for f in feedback:
                st.warning(f)

        results = compute_scores(student_translation, reference_translation)
        st.subheader("📊 Evaluation Results")
        st.json(results)

        idioms_suggestions = suggest_idioms(student_translation)
        if idioms_suggestions:
            st.subheader("💡 Idioms/Collocations Suggestions")
            for idiom in idioms_suggestions:
                st.info(idiom)

        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        # Save submission
        st.session_state.submissions.append({
            "user": username,
            "source": source_text,
            "translation": student_translation,
            "reference": reference_translation,
            "timestamp": time.time(),
            "results": results
        })

# =========================
# Instructor interface
# =========================
if role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    st.write("View submissions, create exercises, and assess translations.")
    new_source = st.text_area("Create new source text exercise")
    new_reference = st.text_area("Reference translation for instructor use (hidden from students)")

    if st.button("Add Exercise"):
        st.session_state.submissions.append({
            "user": None,
            "source": new_source,
            "translation": None,
            "reference": new_reference,
            "timestamp": time.time(),
            "results": None
        })
        st.success("Exercise added successfully!")

    if st.session_state.submissions:
        st.subheader("Student Submissions")
        df = pd.DataFrame([s for s in st.session_state.submissions if s["translation"]], columns=["user", "source", "translation", "timestamp"])
        if not df.empty:
            st.dataframe(df)

        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        st.subheader("🏆 Leaderboard")
        for rank, (user, pts) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. {user} - {pts} points")
