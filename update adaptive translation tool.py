import streamlit as st
import time
import random
import pandas as pd
from difflib import SequenceMatcher

# Metrics
import sacrebleu
from bert_score import score as bert_score
import evaluate

# Load COMET model from Hugging Face Evaluate
try:
    comet = evaluate.load("comet")
except Exception as e:
    comet = None
    st.warning(f"⚠️ COMET not available: {e}")

# ==========================
# Helper Functions
# ==========================

def compute_scores(student_translation, reference):
    results = {}
    if reference:
        # BLEU, chrF, TER
        bleu = sacrebleu.corpus_bleu([student_translation], [[reference]])
        chrf = sacrebleu.corpus_chrf([student_translation], [[reference]])
        ter = sacrebleu.corpus_ter([student_translation], [[reference]])
        results["BLEU"] = round(bleu.score, 2)
        results["chrF"] = round(chrf.score, 2)
        results["TER"] = round(ter.score, 2)

        # Edit distance
        matcher = SequenceMatcher(None, student_translation, reference)
        edit_distance = int((1 - matcher.ratio()) * max(len(student_translation), len(reference)))
        results["Edit Distance"] = edit_distance
    else:
        results["BLEU"] = "N/A"
        results["chrF"] = "N/A"
        results["TER"] = "N/A"
        results["Edit Distance"] = "N/A"

    # BERTScore
    try:
        if reference:
            P, R, F1 = bert_score([student_translation], [reference], lang="en", rescale_with_baseline=True)
            results["BERTScore F1"] = round(float(F1.mean()), 4)
        else:
            results["BERTScore F1"] = "N/A"
    except Exception as e:
        results["BERTScore F1"] = f"Error: {e}"

    # COMET
    if comet and reference:
        try:
            comet_result = comet.compute(predictions=[student_translation], references=[reference], sources=["N/A"])
            results["COMET"] = round(comet_result["mean_score"], 4)
        except Exception as e:
            results["COMET"] = f"COMET Error: {e}"
    else:
        results["COMET"] = "N/A"

    return results

def suggest_idioms():
    idioms = [
        "break the ice",
        "a blessing in disguise",
        "hit the nail on the head",
        "once in a blue moon",
        "spill the beans",
        "kick the bucket",
        "the ball is in your court",
        "burn the midnight oil"
    ]
    return random.sample(idioms, 2)

def categorize_errors(student_translation, reference):
    errors = []
    if not reference:
        return ["No reference provided – error categorization skipped."]
    if student_translation.lower() == reference.lower():
        return ["Perfect match – no errors."]
    if len(student_translation.split()) < len(reference.split()):
        errors.append("Possible omission (student translation is shorter).")
    if len(student_translation.split()) > len(reference.split()):
        errors.append("Possible addition (student translation is longer).")
    if any(word not in reference for word in student_translation.split()):
        errors.append("Possible lexical choice error.")
    return errors if errors else ["Minor stylistic differences."]

# ==========================
# Streamlit App
# ==========================

st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if role == "Student":
    st.subheader("✍️ Submit Your Translation")
    source_text = st.text_area("Source Text", "Enter the text to translate...")
    student_translation = st.text_area("Your Translation")
    reference = st.text_area("Reference Translation (optional, hidden from students)")

    if st.button("Evaluate"):
        start = time.time()
        results = compute_scores(student_translation, reference if reference.strip() else None)
        end = time.time()

        st.subheader("📊 Evaluation Results")
        st.json(results)

        st.write(f"⏱️ Time Taken: {round(end - start, 2)} seconds")
        st.write(f"🏆 Points earned: {random.randint(5,15)}")

        st.subheader("💡 Idioms/Collocations Suggestions")
        for idiom in suggest_idioms():
            st.write(f"- Consider using: **{idiom}**")

        st.subheader("🔎 Error Categorization")
        for err in categorize_errors(student_translation, reference if reference.strip() else None):
            st.write(f"- {err}")

elif role == "Instructor":
    st.subheader("📊 Instructor Dashboard")
    uploaded = st.file_uploader("Upload student submissions CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

        if "Points" in df.columns and "Username" in df.columns:
            leaderboard = df.groupby("Username")["Points"].sum().reset_index().sort_values(by="Points", ascending=False)
            st.write("🏆 Leaderboard")
            st.dataframe(leaderboard)
