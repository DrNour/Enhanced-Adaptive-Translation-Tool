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
    from bert_score import score as bert_score
    bert_available = True
except ModuleNotFoundError:
    bert_available = False

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False

# COMET optional
comet_enabled = False
COMET_TOKEN = st.secrets.get("COMET_TOKEN") if "COMET_TOKEN" in st.secrets else None
if COMET_TOKEN:
    comet_enabled = True

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

# User Role
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# Session State
if "score" not in st.session_state: st.session_state.score = 0
if "leaderboard" not in st.session_state: st.session_state.leaderboard = {}
if "submissions" not in st.session_state: st.session_state.submissions = []
if "idioms" not in st.session_state: st.session_state.idioms = ["break the ice", "once upon a time"]

# =========================
# Functions
# =========================
def update_score(user, points):
    st.session_state.score += points
    st.session_state.leaderboard[user] = st.session_state.leaderboard.get(user,0) + points

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

def evaluate_translation(student_translation, reference_translation=None):
    results = {}

    # BLEU / chrF / TER
    if sacrebleu_available and reference_translation:
        try:
            results["BLEU"] = sacrebleu.corpus_bleu([student_translation],[ [reference_translation] ]).score
            results["chrF"] = sacrebleu.corpus_chrf([student_translation],[ [reference_translation] ]).score
            results["TER"] = sacrebleu.corpus_ter([student_translation],[ [reference_translation] ]).score
        except:
            results["BLEU"] = results["chrF"] = results["TER"] = "Error"
    else:
        results["BLEU"] = results["chrF"] = results["TER"] = "N/A"

    # Edit distance
    if levenshtein_available and reference_translation:
        try:
            results["Edit Distance"] = Levenshtein.distance(student_translation, reference_translation)
        except:
            results["Edit Distance"] = "Error"
    else:
        results["Edit Distance"] = "N/A"

    # BERTScore
    if bert_available:
        try:
            P, R, F1 = bert_score([student_translation],[student_translation if not reference_translation else reference_translation], lang="en", rescale_with_baseline=True)
            results["BERT_F1"] = float(F1.mean())
        except:
            results["BERT_F1"] = "Error"
    else:
        results["BERT_F1"] = "N/A"

    # COMET optional
    if comet_enabled:
        results["COMET"] = "Optional COMET evaluation here (requires API integration)"
    else:
        results["COMET"] = "Disabled"

    return results

# =========================
# Student Interface
# =========================
if role=="Student":
    st.subheader("📝 Submit Your Translation")
    source_text = st.text_area("Source Text", height=150)
    student_translation = st.text_area("Your Translation", height=150)

    if st.button("Submit Translation"):
        start_time = time.time()
        results = evaluate_translation(student_translation)
        elapsed = time.time() - start_time

        # Idioms / collocations
        suggestions = [i for i in st.session_state.idioms if i in student_translation.lower()]
        if not suggestions: suggestions = ["None detected"]

        # Display results
        st.subheader("📊 Evaluation Results")
        st.json(results)
        st.write(f"Time Taken: {elapsed:.2f} seconds")
        points = 10 + int(random.random()*10)
        update_score(username, points)
        st.success(f"Points earned: {points}")

        st.subheader("💡 Idioms / Collocations Suggestions")
        st.write(suggestions)

        # Save submission to instructor
        st.session_state.submissions.append({
            "username": username,
            "source": source_text,
            "translation": student_translation,
            "results": results
        })

# =========================
# Instructor Interface
# =========================
if role=="Instructor":
    st.subheader("📊 Instructor Dashboard")
    st.write("View student submissions:")

    if st.session_state.submissions:
        for i, s in enumerate(st.session_state.submissions):
            st.markdown(f"**Student:** {s['username']}")
            st.markdown(f"**Source:** {s['source']}")
            st.markdown(f"**Translation:** {s['translation']}")
            st.markdown(f"**Results:** {s['results']}")
            st.markdown("---")
    else:
        st.info("No submissions yet.")

    st.subheader("Leaderboard")
    if st.session_state.leaderboard:
        sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        for rank, (user, pts) in enumerate(sorted_lb, start=1):
            st.write(f"{rank}. **{user}** - {pts} points")
    else:
        st.info("No points yet.")

    # Optional charts
    if pd_available and st.session_state.leaderboard:
        df = pd.DataFrame([{"Username": k, "Points": v} for k,v in st.session_state.leaderboard.items()])
        st.dataframe(df)
        st.bar_chart(df.set_index("Username")["Points"])
