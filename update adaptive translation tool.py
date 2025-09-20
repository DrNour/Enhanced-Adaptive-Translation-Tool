import streamlit as st
from difflib import SequenceMatcher
import pandas as pd
import time
import os

# Optional scoring imports
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except:
    SACREBLEU_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

# --- Initialize session state ---
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "keystrokes" not in st.session_state:
    st.session_state.keystrokes = []
if "translations" not in st.session_state:
    st.session_state.translations = []

# --- Role selection ---
role = st.radio("I am a:", ["Student", "Instructor"])
name = st.text_input("Enter your name:")

# --- Functions ---
def record_keystroke():
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
    st.session_state.keystrokes.append(time.time())

def highlight_diff(student, reference):
    student = str(student) if student else ""
    reference = str(reference) if reference else ""
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
    if reference.strip():
        if SACREBLEU_AVAILABLE:
            results["BLEU"] = sacrebleu.corpus_bleu([student], [[reference]]).score
            results["chrF"] = sacrebleu.corpus_chrf([student], [[reference]]).score
            results["TER"] = sacrebleu.corpus_ter([student], [[reference]]).score
        if BERT_AVAILABLE:
            P, R, F1 = bert_score([student], [reference], lang="en")
            results["BERT_F1"] = float(F1.mean())
    return results

# --- Student Interface ---
if role == "Student":
    st.subheader("Submit your translation")
    student_translation = st.text_area("Your Translation:", key="student_input", on_change=record_keystroke)
    reference_translation = ""  # Hide from student

    submit = st.button("Submit")
    if submit:
        time_spent = time.time() - st.session_state.start_time if st.session_state.start_time else 0
        st.write(f"Time spent: {time_spent:.2f} sec")
        st.write(f"Keystrokes: {len(st.session_state.keystrokes)}")

        # Highlight edits only if reference exists
        if reference_translation.strip():
            highlighted_text, feedback = highlight_diff(student_translation, reference_translation)
            st.markdown("### Edits Highlighted")
            st.markdown(highlighted_text, unsafe_allow_html=True)
            st.write("### Feedback")
            for f in feedback:
                st.write(f)

        # Optional scoring
        scores = compute_scores(student_translation, reference_translation)
        if scores:
            st.write("### Scores")
            for k, v in scores.items():
                st.write(f"{k}: {v}")

        # Idioms/collocations placeholder
        st.write("### Idioms/Collocations Suggestions")
        st.write("Feature coming soon...")

        # Save to session / CSV
        st.session_state.translations.append({
            "Name": name,
            "Translation": student_translation,
            "Time_Spent": time_spent,
            "Keystrokes": len(st.session_state.keystrokes)
        })
        df = pd.DataFrame(st.session_state.translations)
        df.to_csv("submissions.csv", index=False)
        st.success("Submission saved!")

# --- Instructor Interface ---
elif role == "Instructor":
    st.subheader("Instructor Dashboard")
    if os.path.exists("submissions.csv"):
        df = pd.read_csv("submissions.csv")
        st.dataframe(df)
    uploaded_file = st.file_uploader("Upload student CSV (optional)")
    if uploaded_file:
        df2 = pd.read_csv(uploaded_file)
        st.dataframe(df2)
