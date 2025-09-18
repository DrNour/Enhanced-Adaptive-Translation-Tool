import streamlit as st
import pandas as pd
import requests
import difflib
import Levenshtein as lev
import sacrebleu
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint

# ------------------------
# Setup
# ------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")

# Initialize COMET models (cached to avoid re-downloads)
@st.cache_resource
def load_comet_models():
    try:
        model_ref = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        model_qe = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        return model_ref, model_qe
    except Exception:
        return None, None

comet_model_ref, comet_model_qe = load_comet_models()

# ------------------------
# Helper Functions
# ------------------------
def translate_text(text, src_lang="en", tgt_lang="ar"):
    """Call Hugging Face Translation API."""
    API_URL = f"https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()[0]["translation_text"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"

def compute_metrics(student, reference=None, source=None):
    """Compute all metrics depending on availability of reference."""
    results = {}

    if reference:  # With reference
        # BLEU / chrF / TER
        bleu = sacrebleu.corpus_bleu([student], [[reference]])
        chrf = sacrebleu.corpus_chrf([student], [[reference]])
        ter = sacrebleu.corpus_ter([student], [[reference]])
        results["BLEU"] = round(bleu.score, 2)
        results["chrF"] = round(chrf.score, 2)
        results["TER"] = round(ter.score, 2)

        # BERTScore
        P, R, F1 = bert_score([student], [reference], lang="en", verbose=False)
        results["BERTScore_F1"] = float(F1.mean())

        # COMET
        if comet_model_ref:
            data = [{"src": source or "", "mt": student, "ref": reference}]
            comet_score = comet_model_ref.predict(data, batch_size=8, gpus=0)
            results["COMET"] = float(comet_score.system_score)
    else:  # Reference-free mode
        if comet_model_qe:
            data = [{"src": source or "", "mt": student}]
            qe_score = comet_model_qe.predict(data, batch_size=8, gpus=0)
            results["COMET-QE"] = float(qe_score.system_score)

    return results

def track_edits(student, reference):
    """Compute edit distance details."""
    if not reference:
        return "No reference provided."
    distance = lev.distance(student, reference)
    ops = lev.opcodes(student, reference)
    return {"EditDistance": distance, "Operations": ops}

# ------------------------
# UI
# ------------------------
st.title("🌍 Adaptive Translation & Post-Editing Tool")

role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if role == "Student":
    st.header("✍️ Submit Translation")

    source = st.text_area("Source Text (English):")
    student_translation = st.text_area("Your Translation:")
    reference = st.text_area("Reference Translation (optional):")

    if st.button("Evaluate"):
        if not student_translation:
            st.warning("Please provide your translation.")
        else:
            metrics = compute_metrics(student_translation, reference, source)
            st.subheader("📊 Evaluation Results")
            st.json(metrics)

            if reference:
                st.subheader("🔍 Edit Tracking")
                edits = track_edits(student_translation, reference)
                st.json(edits)

elif role == "Instructor":
    st.header("📊 Instructor Dashboard")

    uploaded = st.file_uploader("Upload student results CSV (Username, Source, Student, Reference, Points)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        leaderboard = df.groupby("Username")["Points"].sum().reset_index().sort_values(by="Points", ascending=False)
        st.subheader("🏆 Leaderboard")
        st.dataframe(leaderboard)

        st.subheader("📈 Distribution of Points")
        st.bar_chart(leaderboard.set_index("Username"))
