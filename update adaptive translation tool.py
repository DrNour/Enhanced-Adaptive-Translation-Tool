import streamlit as st
from difflib import SequenceMatcher
import time
import random

# Optional packages
try:
    import sacrebleu
    sacrebleu_available = True
except ModuleNotFoundError:
    sacrebleu_available = False
    st.warning("sacrebleu not installed: BLEU/chrF/TER scoring disabled.")

try:
    import Levenshtein
    levenshtein_available = True
except ModuleNotFoundError:
    levenshtein_available = False
    st.warning("python-Levenshtein not installed: edit distance scoring disabled.")

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd_available = True
except ModuleNotFoundError:
    pd_available = False
    st.warning("pandas/seaborn/matplotlib not installed: Instructor dashboard charts disabled.")

# Semantic evaluation
try:
    from bert_score import score as bert_score
    bert_available = True
except ModuleNotFoundError:
    bert_available = False
    st.warning("BERTScore not installed: semantic evaluation disabled.")

try:
    from comet_mt import get_comet_score  # hypothetical function; replace with real COMET call
    comet_available = True
except ModuleNotFoundError:
    comet_available = False
    st.warning("COMET not installed: contextual MT evaluation disabled.")

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# Session states
if "score" not in st.session_state: st.session_state.score = 0
if "leaderboard" not in st.session_state: st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state: st.session_state.feedback_history = []

# =========================
# Leaderboard / Score update
# =========================
def update_score(username, points):
    st.session_state.score += points
    if username not in st.session_state.leaderboard:
        st.session_state.leaderboard[username] = 0
    st.session_state.leaderboard[username] += points

# =========================
# Error highlighting
# =========================
def highlight_diff(student
