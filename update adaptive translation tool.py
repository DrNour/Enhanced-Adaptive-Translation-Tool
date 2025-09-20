import streamlit as st
import pandas as pd
import time
import sacrebleu
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# =====================
# Setup
# =====================
nltk.download("punkt_tab", quiet=True)

@st.cache_resource
def load_semantic_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, semantic_model = load_semantic_model()

# =====================
# Metrics
# =====================
def compute_bleu(student_text, reference_text):
    return sacrebleu.corpus_bleu([student_text], [[reference_text]]).score

def compute_semantic(student_text, reference_text, tokenizer, model):
    def embed(text):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**tokens)
        return output.last_hidden_state.mean(dim=1)

    emb1 = embed(student_text)
    emb2 = embed(reference_text)
    sim = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
    return sim * 100

def compute_fluency(student_text):
    words = nltk.word_tokenize(student_text)
    avg_len = sum(len(w) for w in words) / (len(words) + 1e-5)
    return max(0, 100 - (avg_len - 5) * 10)

# =====================
# Sidebar Menu
# =====================
st.sidebar.title("Adaptive Translation Tool")
mode = st.sidebar.radio("Choose mode:", ["Instructor", "Student"])

# =====================
# Data storage
# =====================
if "exercises" not in st.session_state:
    st.session_state["exercises"] = {}

if "results" not in st.session_state:
    st.session_state["results"] = []

# =====================
# Instructor Mode
# =====================
if mode == "Instructor":
    st.header("Instructor Dashboard")
    ex_name = st.text_input("Exercise Name")
    ref_text = st.text_area("Reference Translation")

    if st.button("Assign Exercise"):
        if ex_name and ref_text:
            st.session_state["exercises"][ex_name] = ref_text
            st.success(f"Exercise '{ex_name}' created.")
        else:
            st.error("Please provide both a name and reference translation.")

    if st.session_state["exercises"]:
        st.subheader("Assigned Exercises")
        for name, ref in st.session_state["exercises"].items():
            st.write(f"**{name}** → {ref}")

    if st.session_state["results"]:
        st.subheader("Student Results")
        df = pd.DataFrame(st.session_state["results"])
        st.dataframe(df)

# =====================
# Student Mode
# =====================
if mode == "Student":
    st.header("Student Dashboard")

    if not st.session_state["exercises"]:
        st.info("No exercises available yet. Please wait for your instructor.")
    else:
        choice = st.selectbox("Choose an exercise:", list(st.session_state["exercises"].keys()))
        reference = st.session_state["exercises"][choice]

        st.write("### Reference Translation")
        st.write(reference)

        st.write("### Your Translation / Edited Text")
        start_time = time.time()
        student_text = st.text_area("Edit and improve the translation here:")
        keystrokes = len(student_text)

        if st.button("Submit"):
            end_time = time.time()
            time_spent = round(end_time - start_time, 2)

            bleu = compute_bleu(student_text, reference)
            semantic = compute_semantic(student_text, reference, tokenizer, semantic_model)
            fluency = compute_fluency(student_text)

            result = {
                "Exercise": choice,
                "Student Text": student_text,
                "BLEU": round(bleu, 2),
                "Semantic Accuracy": round(semantic, 2),
                "Fluency": round(fluency, 2),
                "Keystrokes": keystrokes,
                "Time Spent (s)": time_spent,
            }

            st.session_state["results"].append(result)

            st.success("Submission recorded!")
            st.json(result)
