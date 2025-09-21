import streamlit as st
import sqlite3
import time
import sacrebleu
from datetime import datetime
import difflib

# Optional heavy import (wrapped in try/except so app won't crash)
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# ============ DATABASE ============
def init_db():
    conn = sqlite3.connect("translations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            mt_output TEXT,
            reference TEXT,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exercise_id INTEGER,
            student_name TEXT,
            student_edit TEXT,
            time_spent REAL,
            keystrokes INTEGER,
            bleu REAL,
            chrf REAL,
            ter REAL,
            bert_f1 REAL,
            similarity REAL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============ SCORING ============
def compute_scores(hypothesis, reference, use_bert=True):
    results = {}
    if not reference.strip():
        return {"BLEU": None, "chrF": None, "TER": None, "BERT_F1": None, "similarity": None}

    try:
        results["BLEU"] = round(sacrebleu.corpus_bleu([hypothesis], [[reference]]).score, 3)
    except:
        results["BLEU"] = None

    try:
        results["chrF"] = round(sacrebleu.corpus_chrf([hypothesis], [[reference]]).score, 3)
    except:
        results["chrF"] = None

    try:
        results["TER"] = round(sacrebleu.corpus_ter([hypothesis], [[reference]]).score, 3)
    except:
        results["TER"] = None

    # Fallback: difflib similarity
    try:
        results["similarity"] = round(difflib.SequenceMatcher(None, hypothesis, reference).ratio(), 3)
    except:
        results["similarity"] = None

    # BERTScore (optional)
    if use_bert and BERT_AVAILABLE:
        try:
            _, _, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=True)
            results["BERT_F1"] = float(round(F1[0].item(), 3))
        except:
            results["BERT_F1"] = None
    else:
        results["BERT_F1"] = None

    return results

# ============ APP ============
st.sidebar.title("Navigation")
role = st.sidebar.selectbox("I am a", ["Student", "Instructor"])

if role == "Instructor":
    st.title("📚 Instructor Dashboard")
    menu = st.sidebar.radio("Choose Action", ["Create Exercise", "View Submissions"])

    if menu == "Create Exercise":
        st.subheader("Create a New Exercise")
        source = st.text_area("Source Text")
        mt_output = st.text_area("Machine Translation Output")
        reference = st.text_area("Reference Translation (optional)")
        instructor = st.text_input("Instructor Name")

        if st.button("Save Exercise"):
            conn = sqlite3.connect("translations.db")
            c = conn.cursor()
            c.execute("INSERT INTO exercises (source, mt_output, reference, created_by) VALUES (?, ?, ?, ?)",
                      (source, mt_output, reference, instructor))
            conn.commit()
            conn.close()
            st.success("✅ Exercise created successfully!")

    elif menu == "View Submissions":
        st.subheader("Student Submissions")
        conn = sqlite3.connect("translations.db")
        c = conn.cursor()
        c.execute("""
            SELECT s.id, e.source, e.mt_output, e.reference, s.student_name,
                   s.student_edit, s.bleu, s.chrf, s.ter, s.bert_f1, s.similarity,
                   s.time_spent, s.keystrokes, s.submitted_at
            FROM submissions s
            JOIN exercises e ON s.exercise_id = e.id
            ORDER BY s.submitted_at DESC
        """)
        rows = c.fetchall()
        conn.close()

        for r in rows:
            st.markdown(f"""
            **Student:** {r[4]}  
            **Submitted At:** {r[13]}  
            **Source:** {r[1]}  
            **MT Output:** {r[2]}  
            **Student Edit:** {r[5]}  
            **Reference:** {r[3]}  

            📊 **Scores**  
            - BLEU: {r[6]}  
            - chrF: {r[7]}  
            - TER: {r[8]}  
            - BERT F1: {r[9]}  
            - Similarity: {r[10]}  

            ⌛ **Time Spent:** {r[11]} sec  
            ⌨️ **Keystrokes:** {r[12]}  
            """)
            st.markdown("---")

elif role == "Student":
    st.title("✍️ Student Exercise")
    student = st.text_input("Enter your name")

    conn = sqlite3.connect("translations.db")
    c = conn.cursor()
    c.execute("SELECT id, source, mt_output, reference FROM exercises ORDER BY created_at DESC")
    exercises = c.fetchall()
    conn.close()

    if not exercises:
        st.warning("⚠️ No exercises available yet. Please wait for your instructor.")
    else:
        choice = st.selectbox("Choose an Exercise", [f"Exercise {e[0]}" for e in exercises])
        selected = exercises[int(choice.split()[1]) - 1]

        st.markdown(f"**Source Text:** {selected[1]}")
        st.markdown(f"**Machine Translation Output:** {selected[2]}")
        reference = selected[3]

        start_time = time.time()
        student_edit = st.text_area("Edit the Translation Here ✍️", value=selected[2])
        keystrokes = len(student_edit)

        use_bert = st.checkbox("Enable BERTScore (slower, optional)", value=False)

        if st.button("Submit"):
            end_time = time.time()
            time_spent = round(end_time - start_time, 2)
            scores = compute_scores(student_edit, reference or "", use_bert=use_bert)

            conn = sqlite3.connect("translations.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO submissions (exercise_id, student_name, student_edit,
                                         time_spent, keystrokes, bleu, chrf, ter, bert_f1, similarity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (selected[0], student, student_edit, time_spent, keystrokes,
                  scores.get("BLEU"), scores.get("chrF"), scores.get("TER"),
                  scores.get("BERT_F1"), scores.get("similarity")))
            conn.commit()
            conn.close()

            st.success("✅ Submission saved and evaluated!")
            st.json(scores)
