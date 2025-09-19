import streamlit as st
import random
import time
from difflib import SequenceMatcher
import json

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
    pd_available = True
except ModuleNotFoundError:
    pd_available = False

try:
    from bert_score import score as bert_score
    bert_available = True
except ModuleNotFoundError:
    bert_available = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    hf_available = True
except ModuleNotFoundError:
    hf_available = False

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Enhanced Adaptive Translation Tool", layout="wide")
st.title("🌍 Enhanced Adaptive Translation & Post-Editing Tool")

# Session state initialization
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "submissions" not in st.session_state:
    st.session_state.submissions = []
if "exercises" not in st.session_state:
    st.session_state.exercises = []

# =========================
# Role Selection
# =========================
role = st.radio("I am a:", ["Student", "Instructor"], index=0)
st.session_state.role = role
username = st.text_input("Enter your name:", value=st.session_state.username)
st.session_state.username = username

# =========================
# Helper Functions
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

def compute_scores(student, reference):
    results = {}
    if reference:
        if sacrebleu_available:
            results['BLEU'] = sacrebleu.corpus_bleu([student], [[reference]]).score
            results['chrF'] = sacrebleu.corpus_chrf([student], [[reference]]).score
            results['TER'] = sacrebleu.corpus_ter([student], [[reference]]).score
        if levenshtein_available:
            results['Edit_Distance'] = Levenshtein.distance(student, reference)
    return results

def compute_bert(student, reference=None):
    if bert_available:
        if reference:
            P, R, F1 = bert_score([student], [reference], lang="en", verbose=False)
        else:
            P, R, F1 = bert_score([student], [student], lang="en", verbose=False)
        return float(F1.mean())
    else:
        return "BERTScore not available"

def comet_score(student, reference=None):
    if hf_available and "HF_TOKEN" in st.secrets:
        import requests
        headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        data = {"src": "dummy", "mt": student}
        if reference:
            data["ref"] = reference
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/UNITEComet/comet-qe-msmarco",
                headers=headers, json=data
            )
            if response.status_code == 200:
                return response.json()
            else:
                return f"COMET Error {response.status_code}"
        except Exception as e:
            return str(e)
    else:
        return "COMET not available"

# =========================
# Instructor Dashboard
# =========================
if role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    
    # Add exercise
    with st.expander("Add New Exercise"):
        src_text = st.text_area("Source Text")
        ref_text = st.text_area("Reference Translation (Optional)")
        idioms = st.text_area("Idioms/Collocations (Optional, comma-separated)")
        if st.button("Add Exercise"):
            exercise = {
                "source": src_text,
                "reference": ref_text,
                "idioms": [i.strip() for i in idioms.split(",") if i.strip()]
            }
            st.session_state.exercises.append(exercise)
            st.success("Exercise added!")

    # View student submissions
    st.subheader("Student Submissions")
    if st.session_state.submissions:
        for sub in st.session_state.submissions:
            st.markdown(f"**{sub['student']}** submitted for exercise {sub['exercise_id']}:")
            st.write(sub['translation'])
            st.json(sub['results'])
            if sub.get('feedback'):
                st.write("Feedback:", sub['feedback'])
    else:
        st.info("No submissions yet.")
    
    # Export CSV
    if pd_available and st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        st.download_button("Export CSV", df.to_csv(index=False), "submissions.csv", "text/csv")

# =========================
# Student Interface
# =========================
else:
    st.subheader("📝 Translate & Submit")
    
    if st.session_state.exercises:
        exercise = random.choice(st.session_state.exercises)
        st.markdown(f"**Source Text:**\n{exercise['source']}")
        student_translation = st.text_area("Your Translation", height=150)
        
        if st.button("Submit Translation"):
            results = compute_scores(student_translation, exercise.get('reference'))
            results['BERT_F1'] = compute_bert(student_translation, exercise.get('reference'))
            results['COMET'] = comet_score(student_translation, exercise.get('reference'))

            feedback_text = []
            if exercise.get('reference'):
                _, feedback_text = highlight_diff(student_translation, exercise['reference'])
            st.subheader("📊 Evaluation Results")
            st.json(results)
            if feedback_text:
                st.write("💡 Feedback:")
                for f in feedback_text:
                    st.write("-", f)

            if exercise.get('idioms'):
                st.write("💡 Idioms/Collocations Suggestions:")
                for idiom in exercise['idioms']:
                    st.write("-", idiom)
            
            # Save submission for instructor
            submission = {
                "student": username,
                "exercise_id": st.session_state.exercises.index(exercise),
                "translation": student_translation,
                "results": results,
                "feedback": feedback_text
            }
            st.session_state.submissions.append(submission)
    else:
        st.info("No exercises available. Please wait for the instructor to add exercises.")
