import os
import streamlit as st
from bert_score import score as bert_score

# Optional COMET
try:
    from comet import download_model, load_from_checkpoint
    comet_available = True
except ImportError:
    comet_available = False

HF_TOKEN = os.getenv("HF_TOKEN", None)

def evaluate_translation(src, mt, ref=None, use_comet=False):
    results = {}

    # BERTScore
    P, R, F1 = bert_score([mt], [ref if ref else src], lang="en", verbose=False)
    results["BERT_F1"] = round(F1.mean().item(), 4)

    # Optional COMET
    if use_comet and comet_available and HF_TOKEN:
        try:
            model_path = download_model("wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            data = [{"src": src, "mt": mt, "ref": ref if ref else src}]
            seg_scores, sys_score = model.predict(data, batch_size=8, gpus
