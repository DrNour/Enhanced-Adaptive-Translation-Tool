import streamlit as st
import time
import random
import difflib
import pandas as pd
import csv
from datetime import datetime

# =========================
# Optional packages
# =========================
try:
    import sacrebleu
    sacrebleu_available = True
except ImportError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER disabled.")

try:
    from bert_score import score as bertscore
    bertscore_available = True
except ImportError:
    bertscore_available = False
    st.warning("bert-score not installed: BERTScore disabled.")

try:
    from comet import download_model, load_from_checkpoint
    comet_model_path = download_model("Unbabel/wmt20-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_available = True
except Exception:
    comet_available = False
    st.warning("COMET not installed or model not loaded: COMET disabled.")

# =========================
# Helper Functions
# =========================
def calculate_metrics(hypothesis, reference, source=None):
    results = {}
    if sacrebleu_available:
        results["BLEU"] = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score
        results["chrF"] = sacrebleu.corpus_chrf([hypothesis], [[reference]]).score
        results["TER"] = sacrebleu.corpus_ter([hypothesis], [[reference]]).score
    if bertscore_available:
        P, R, F1 = bertscore([hypothesis], [reference], lang="en")
        results["BERTScore"] = float(F1.mean())
    if comet_available and source is not None:
        data = [{"src": source, "mt": hypothesis, "ref": reference}]
        results["COMET"] = comet_model.predict(data, batch_size=1, gpus=0)["scores"][0]
    return results

def highlight_diff(student, reference):
    matcher = difflib.SequenceMatcher(None, reference.split(), student.split())
    highlighted, feedback = "", []
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

def detect_domain(text):
    text = text.lower()
    if any(w in text for w in ["court", "law", "contract", "agreement"]):
        return "Legal"
    elif any(w in text for w in ["king", "love", "poem", "story"]):
        return "Literary"
    elif any(w in text for w in ["breaking", "news", "journalist", "headline"]):
        return "Journalistic"
    else:
        return "General"

def log_progress(username, role, source, reference, student, scores, feedback, elapsed):
    log_entry = {
        "Timestamp": datetime.now().isoformat(),
        "Username": username,
        "Role": role,
        "Source": source,
        "Reference": reference,
        "StudentTranslation": student,
        "Scores": scores,
        "Feedback": feedback,
        "TimeTaken": elapsed,
    }
    file_exists = False
    try:
        with open("progress_log.csv", "r", encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        pass
    with open("progress_log.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if role == "Student":
    st.subheader("📝 Submit Your Translation")
    source_text = st.text_area("Source Text")
    student_translation = st.text_area("Your Translation", height=150)

    reference_translation = None
    if st.checkbox("I have a reference translation (optional)"):
        reference_translation = st.text_area("Reference Translation (hidden from others)")

    if st.button("Evaluate Translation"):
        start_time = time.time()
        domain = detect_domain(source_text)
        st.info(f"Detected domain: **{domain}**")

        if reference_translation:
            highlighted, fb = highlight_diff(student_translation, reference_translation)
            st.markdown(highlighted, unsafe_allow_html=True)

            scores = calculate_metrics(student_translation, reference_translation, source_text)
            for k, v in scores.items():
                st.write(f"{k}: {v:.2f}")

            st.subheader("Feedback")
            for f in fb:
                st.warning(f)
        else:
            scores, fb = {}, ["No reference provided → semantic evaluation skipped. Focus on fluency/coherence."]
            st.info(fb[0])

        elapsed_time = time.time() - start_time
        st.success(f"Time Taken: {elapsed_time:.2f} seconds")

        log_progress(username, role, source_text, reference_translation, student_translation, scores, fb, elapsed_time)

elif role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    try:
        df = pd.read_csv("progress_log.csv")
        st.dataframe(df.tail(20))
        leaderboard = df.groupby("Username")["Scores"].count().sort_values(ascending=False)
        st.bar_chart(leaderboard)
    except FileNotFoundError:
        st.info("No student submissions yet.")
