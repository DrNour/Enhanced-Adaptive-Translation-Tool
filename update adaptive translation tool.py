import streamlit as st
import sqlite3
import time
import difflib
import sacrebleu
from bert_score import score as bert_score
from datetime import datetime

# ============ DATABASE SETUP ============
def init_db():
    conn = sqlite3.connect("translations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS editing_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            mt_output TEXT,
            reference TEXT,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS editing_submissions (
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
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============ UTILS ============
def compute_scores(hypothesis, reference):
    """Compute BLEU, chrF, TER, and BERTScore."""
    results = {}
    if reference.strip():
        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score
        chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]]).score
        ter = sacrebleu.corpus_ter([hypothesis], [[reference]]).score
        results.update({"BLEU": bleu, "chrF": chrf, "TER": ter})
    else:
        results.update({"BLEU": None, "chrF": None, "TER": None})

    try:
        P, R, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=True)
        results["BERT_F1"] = float(F1[0])
    except Exception as e:
        results["BERT_F1"] = None

    return results

# ============ APP ============
st.sidebar.title("Navigation")
role = st.sidebar.selectbox("I am a", ["Student", "Instructor"])

if role == "Instructor":
    st.title("📚 Instructor Dashboard")

    menu = st.sidebar.radio("Choose Action", ["Create Editing Exercise", "View Submissions"])

    if menu == "Create Editing Exercise":
        st.subheader("Create a New Editing Exercise")
        source = st.text_area("Source Text")
        mt_output = st.text_area("Machine Translation Output")
        reference = st.text_area("Reference Translation (optional)")
        instructor = st.text_input("Instructor Name")

        if st.button("Save Exercise"):
            conn = sqlite3.connect("translations.db")
            c = conn.cursor()
            c.execute("INSERT INTO editing_exercises (source, mt_output, reference, created_by) VALUES (?, ?, ?, ?)",
                      (source, mt_output, reference, instructor))
            conn.commit()
            conn.close()
            st.success("✅ Exercise created successfully!")

    elif menu == "View Submissions":
        st.subheader("Student Submissions")
        conn = sqlite3.connect("translations.db")
        c = conn.cursor()
        c.execute("""
            SELECT es.id, e.source, e.mt_output, e.reference, es.student_name,
                   es.student_edit, es.bleu, es.chrf, es.ter, es.bert_f1, es.time_spent, es.keystrokes, es.submitted_at
            FROM editing_submissions es
            JOIN editing_exercises e ON es.exercise_id = e.id
            ORDER BY es.submitted_at DESC
        """)
        rows = c.fetchall()
        conn.close()

        for r in rows:
            st.markdown(f"""
            **Student:** {r[4]}  
            **Submitted At:** {r[12]}  
            **Source:** {r[1]}  
            **MT Output:** {r[2]}  
            **Student Edit:** {r[5]}  
            **Reference:** {r[3]}  

            📊 **Scores**  
            - BLEU: {r[6]}  
            - chrF: {r[7]}  
            - TER: {r[8]}  
            - BERT F1: {r[9]}  

            ⌛ **Time Spent:** {r[10]} sec  
            ⌨️ **Keystrokes:** {r[11]}  
            """)
            st.markdown("---")

elif role == "Student":
    st.title("✍️ Student Editing Exercise")

    student = st.text_input("Enter your name")
    conn = sqlite3.connect("translations.db")
    c = conn.cursor()
    c.execute("SELECT id, source, mt_output, reference FROM editing_exercises ORDER BY created_at DESC")
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

        # Start tracking edits
        start_time = time.time()
        student_edit = st.text_area("Edit the Translation Here ✍️", value=selected[2])
        keystrokes = len(student_edit)

        if st.button("Submit"):
            end_time = time.time()
            time_spent = round(end_time - start_time, 2)

            scores = compute_scores(student_edit, reference or "")

            conn = sqlite3.connect("translations.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO editing_submissions (exercise_id, student_name, student_edit,
                                                 time_spent, keystrokes, bleu, chrf, ter, bert_f1)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (selected[0], student, student_edit, time_spent, keystrokes,
                  scores.get("BLEU"), scores.get("chrF"), scores.get("TER"), scores.get("BERT_F1")))
            conn.commit()
            conn.close()

            st.success("✅ Submission saved and evaluated!")
            st.json(scores)




