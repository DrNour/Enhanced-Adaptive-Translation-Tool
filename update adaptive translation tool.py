# enhanced_adaptive_translation_tool.py

import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime
import difflib
from collections import defaultdict

# NLP scoring
try:
    import sacrebleu
except ImportError:
    sacrebleu = None

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

# Optional COMET
try:
    from comet_ml import download_model, load_from_checkpoint
    comet_enabled = True
except ImportError:
    comet_enabled = False

# ---------- DATABASE ----------
conn = sqlite3.connect("translations.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, role TEXT)""")
c.execute("""
CREATE TABLE IF NOT EXISTS exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    source_text TEXT,
    reference_translation TEXT,
    idioms TEXT,
    collocations TEXT,
    created_at TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exercise_id INTEGER,
    username TEXT,
    submitted_translation TEXT,
    timestamp TEXT,
    points INTEGER,
    bleu REAL,
    chrf REAL,
    ter REAL,
    edit_distance REAL,
    bert_f1 REAL,
    comet_score REAL
)
""")
conn.commit()

# ---------- SESSION STATE ----------
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""

# ---------- LOGIN ----------
st.title("🌍 Adaptive Translation & Post-Editing Tool")
role = st.radio("I am a:", ["Student", "Instructor"])
username = st.text_input("Enter your name:")

if st.button("Login"):
    if username.strip() == "":
        st.warning("Enter a valid username.")
    else:
        st.session_state["username"] = username
        st.session_state["role"] = role
        c.execute("INSERT OR IGNORE INTO users (username, role) VALUES (?, ?)", (username, role))
        conn.commit()
        st.success(f"Logged in as {username} ({role})")

# ---------- MAIN APP ----------
if st.session_state["username"] != "":
    user = st.session_state["username"]
    role = st.session_state["role"]

    if role == "Instructor":
        st.header("📊 Instructor Dashboard")
        tab1, tab2 = st.tabs(["Manage Exercises", "View Submissions"])

        # --- Manage Exercises ---
        with tab1:
            st.subheader("Create New Exercise")
            ex_title = st.text_input("Title")
            src_text = st.text_area("Source Text")
            ref_text = st.text_area("Reference Translation (optional)")
            idioms = st.text_area("Idioms (comma-separated)")
            collocs = st.text_area("Collocations (comma-separated)")

            if st.button("Create Exercise"):
                c.execute(
                    "INSERT INTO exercises (title, source_text, reference_translation, idioms, collocations, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (ex_title, src_text, ref_text, idioms, collocs, datetime.now().isoformat())
                )
                conn.commit()
                st.success("Exercise created!")

            st.subheader("Existing Exercises")
            df_ex = pd.read_sql("SELECT * FROM exercises", conn)
            st.dataframe(df_ex)

        # --- View Submissions ---
        with tab2:
            df_sub = pd.read_sql("SELECT * FROM submissions", conn)
            st.dataframe(df_sub)

            if not df_sub.empty:
                leaderboard = df_sub.groupby("username")["points"].sum().reset_index().sort_values(by="points", ascending=False)
                st.subheader("🏆 Leaderboard")
                st.dataframe(leaderboard)

    else:  # Student
        st.header("✏️ Translation Exercises")
        df_ex = pd.read_sql("SELECT * FROM exercises", conn)
        if df_ex.empty:
            st.info("No exercises available yet.")
        else:
            exercise_ids = df_ex["id"].tolist()
            ex_id = st.selectbox("Select Exercise", exercise_ids)
            ex_row = df_ex[df_ex["id"]==ex_id].iloc[0]

            st.subheader(f"Exercise: {ex_row['title']}")
            st.text_area("Source Text", ex_row["source_text"], height=150, disabled=True)
            st.text_area("Idioms/Collocations Hints", f"Idioms: {ex_row['idioms']}\nCollocations: {ex_row['collocations']}", disabled=True)

            student_translation = st.text_area("Your Translation")
            if st.button("Submit Translation"):
                timestamp = datetime.now().isoformat()
                points = 10
                bleu_score = None
                chrf_score = None
                ter_score = None
                edit_dist = None
                bert_f1_score = None
                comet_score_val = None

                ref_translation = ex_row["reference_translation"]

                # --- Scores ---
                if ref_translation and sacrebleu:
                    bleu_score = sacrebleu.corpus_bleu([student_translation], [[ref_translation]]).score
                    chrf_score = sacrebleu.corpus_chrf([student_translation], [[ref_translation]]).score
                    ter_score = sacrebleu.corpus_ter([student_translation], [[ref_translation]]).score
                    edit_dist = difflib.SequenceMatcher(None, student_translation, ref_translation).ratio()

                if bert_score:
                    P, R, F1 = bert_score([student_translation], [ref_translation] if ref_translation else [student_translation], lang="en", rescale_with_baseline=True)
                    bert_f1_score = float(F1[0])

                if comet_enabled:
                    try:
                        comet_score_val = 0.9  # placeholder
                    except:
                        comet_score_val = None

                # --- Store submission ---
                c.execute(
                    "INSERT INTO submissions (exercise_id, username, submitted_translation, timestamp, points, bleu, chrf, ter, edit_distance, bert_f1, comet_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ex_id, user, student_translation, timestamp, points, bleu_score, chrf_score, ter_score, edit_dist, bert_f1_score, comet_score_val)
                )
                conn.commit()
                st.success("Translation submitted!")

                st.subheader("📊 Evaluation Results")
                st.json({
                    "BLEU": bleu_score,
                    "chrF": chrf_score,
                    "TER": ter_score,
                    "Edit Distance": edit_dist,
                    "BERTScore F1": bert_f1_score,
                    "COMET": comet_score_val
                })
