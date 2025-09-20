import streamlit as st
import time
import random
import csv
from difflib import SequenceMatcher

# Optional imports
try:
    import sacrebleu
    sacrebleu_available = True
except:
    sacrebleu_available = False

try:
    import Levenshtein
    levenshtein_available = True
except:
    levenshtein_available = False

try:
    from bert_score import score as bert_score
    bert_available = True
except:
    bert_available = False

try:
    from comet_ml import download_model, load_from_checkpoint
    comet_available = True
except:
    comet_available = False

import pandas as pd

# =======================
# App Setup
# =======================
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool")

user_type = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

# =======================
# Session state
# =======================
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "submissions" not in st.session_state:
    st.session_state.submissions = []

# =======================
# Functions
# =======================
def update_score(user, points):
    if user not in st.session_state.leaderboard:
        st.session_state.leaderboard[user] = 0
    st.session_state.leaderboard[user] += points

def highlight_diff(student, reference):
    matcher = SequenceMatcher(None, reference.split(), student.split())
    highlighted = ""
    feedback = []
    for tag, i1, i2, j1
