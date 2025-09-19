import os
import streamlit as st
from bert_score import score as bert_score

# Try importing COMET safely
try:
    from comet import download_model, load_from_checkpoint
    comet_available = True
except ImportError:
    comet_available = False

HF_TOKEN = os.getenv("HF_TOKEN", None)

def evaluate_translation(src, mt, ref=None, use_comet=False):
    results = {}

    # ✅ BERTScore
    try:
        P, R, F1 = bert_score([mt], [ref if ref else src], lang="en", verbose=False)
        results["BERT_F1"] = round(F1.mean().item(), 4)
    except Exception as e:
        results["BERT_F1"] = f"Error: {e}"

    # ✅ COMET (optional)
    if use_comet and comet_available and HF_TOKEN:
        try:
            model_path = download_model("wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            data = [{"src": src, "mt": mt, "ref": ref if ref else src}]
            seg_scores, sys_score = model.predict(data, gpus=0)
            results["COMET"] = round(sys_score, 4)
        except Exception as e:
            results["COMET"] = f"COMET Error: {e}"
    else:
        results["COMET"] = "Skipped (disabled or missing token)"

    return results

# ---------------- Streamlit UI ----------------
st.title("Enhanced Translation Evaluation")

src = st.text_area("Source Text")
mt = st.text_area("Machine Translation")
ref = st.text_area("Reference Translation (optional)")
use_comet = st.checkbox("Enable COMET evaluation (requires HF token)", value=False)

if st.button("Evaluate"):
    if not src or not mt:
        st.warning("⚠️ Please enter both source and translation text.")
    else:
        results = evaluate_translation(src, mt, ref, use_comet)
        st.subheader("📊 Evaluation Results")
        st.json(results)
