import streamlit as st
import requests
import textwrap
import docx
import PyPDF2
import os

# =========================
# Hugging Face API Settings
# =========================
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ar"
API_KEY = os.getenv("HF_API_KEY")  # safer than hardcoding
headers = {"Authorization": f"Bearer {API_KEY}"}

# =========================
# Hugging Face Query
# =========================
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# =========================
# Translation Function
# =========================
def translate_long_text(text, max_chunk_length=500):
    chunks = textwrap.wrap(text, max_chunk_length)
    translated_chunks = []

    for chunk in chunks:
        output = query({"inputs": chunk})
        if isinstance(output, list) and "translation_text" in output[0]:
            translated_chunks.append(output[0]["translation_text"])
        else:
            translated_chunks.append("[ERROR: Translation failed]")
    
    return " ".join(translated_chunks)

# =========================
# File Readers
# =========================
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

uploaded_file = st.file_uploader("Upload a file (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("📄 Extracted English Text")
    st.text_area("Original", text, height=200)

    if st.button("🚀 Translate to Arabic"):
        with st.spinner("Translating... please wait"):
            arabic_translation = translate_long_text(text)

        st.subheader("🕌 Arabic Translation")
        st.text_area("Translation", arabic_translation, height=200)

        # Save to file
        output_file = "translated_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(arabic_translation)

        st.download_button(
            label="⬇️ Download Translation",
            data=arabic_translation,
            file_name="translated_output.txt",
            mime="text/plain",
        )
