import streamlit as st
import sqlite3
import time
import difflib
import sacrebleu
from bert_score import score as bert_score
from datetime import datetime
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)

# ================= DATABASE SETUP =================
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
            meteor REAL,
            chrf REAL,
            ter REAL,
            bert_f1 REAL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= UTILS =================
def compute_scores(hypothesis, reference):
    results = {}
    try:
        # BLEU
        smoothie = SmoothingFunction().method4
        results["BLEU"] = round(sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie), 3)
    except:
        results["BLEU"] = None
    try:
        # METEOR
        results["METEOR"] = round(meteor_score([reference.split()], hypothesis.split()), 3)
    except:
        results["METEOR"] = None
    try:
        # chrF & TER
        results["chrF"] = round(sacrebleu.corpus_chrf([hypothesis], [[reference]]).score, 3)
        results["TER"] = round(sacrebleu.corpus_ter([hypothesis], [[reference]]).score, 3)
    except:
        results["chrF"], results["TER"] = None, None
    try:
        # BERTScore
        P, R, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=True)
        results["BERT_F1"] = round(float(F1[0]), 3)
    except:
        results["BERT_F1"] = None
    return results

# ================= APP =================
st.sidebar.title("Navigation")
role = st.sidebar.selectbox("I am a", ["Student", "Instructor"])

if role == "Instructor":
    st.title("📊 Instructor Dashboard")
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
            st.success("✅ Exercise saved!")

    elif menu == "View Submissions":
        st.subheader("All Student Submissions")
        conn = sqlite3.connect("translations.db")
        c = conn.cursor()
        c.execute("""
            SELECT s.id, e.source, e.mt_output, e.reference, s.student_name, s.student_edit,
                   s.bleu, s.meteor, s.chrf, s.ter, s.bert_f1, s.time_spent, s.keystrokes, s.submitted_at
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

📊 **Scores**  
- BLEU: {r[6]}  
- METEOR: {r[7]}  
- chrF: {r[8]}  
- TER: {r[9]}  
- BERT F1: {r[10]}  

⌛ Time Spent: {r[11]} sec  
⌨️ Keystrokes: {r[12]}  
""")
            st.markdown("---")

elif role == "Student":
    st.title("✍️ Student Editing Exercise")
    student = st.text_input("Enter Your Name")
    conn = sqlite3.connect("translations.db")
    c = conn.cursor()
    c.execute("SELECT id, source, mt_output, reference FROM exercises ORDER BY created_at DESC")
    exercises = c.fetchall()
    conn.close()

    if not exercises:
        st.warning("⚠️ No exercises available yet.")
    else:
        choice = st.selectbox("Choose an Exercise", [f"Exercise {e[0]}" for e in exercises])
        selected = exercises[int(choice.split()[1]) - 1]

        st.markdown(f"**Source Text:** {selected[1]}")
        st.markdown(f"**Machine Translation Output:** {selected[2]}")
        reference = selected[3]

        # Track keystrokes and time
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()
        student_edit = st.text_area("Edit the Translation Here ✍️", value=selected[2])
        keystrokes = len(student_edit)

        if st.button("Submit"):
            time_spent = round(time.time() - st.session_state.start_time, 2)
            scores = compute_scores(student_edit, reference or "")
            conn = sqlite3.connect("translations.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO submissions (exercise_id, student_name, student_edit,
                                         time_spent, keystrokes, bleu, meteor, chrf, ter, bert_f1)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (selected[0], student, student_edit, time_spent, keystrokes,
                  scores.get("BLEU"), scores.get("METEOR"), scores.get("chrF"), scores.get("TER"), scores.get("BERT_F1")))
            conn.commit()
            conn.close()
            st.success("✅ Submission saved!")
            st.json(scores)
