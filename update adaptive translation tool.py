import streamlit as st
import sqlite3
import time
from datetime import datetime

# -----------------------------
# DB Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect("exercises.db")
    cur = conn.cursor()

    # Exercises table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        source_text TEXT,
        ref_translation TEXT
    )
    """)

    # Submissions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exercise_id INTEGER,
        student_name TEXT,
        submitted_translation TEXT,
        time_spent REAL,
        keystrokes INTEGER,
        submitted_at TEXT,
        FOREIGN KEY(exercise_id) REFERENCES exercises(id)
    )
    """)

    conn.commit()
    conn.close()

# -----------------------------
# DB Functions
# -----------------------------
def add_exercise(title, source_text, ref_translation):
    conn = sqlite3.connect("exercises.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO exercises (title, source_text, ref_translation) VALUES (?, ?, ?)",
                (title, source_text, ref_translation))
    conn.commit()
    conn.close()

def get_exercises():
    conn = sqlite3.connect("exercises.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM exercises")
    rows = cur.fetchall()
    conn.close()
    return rows

def add_submission(exercise_id, student_name, submitted_translation, time_spent, keystrokes):
    conn = sqlite3.connect("exercises.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO submissions (exercise_id, student_name, submitted_translation, time_spent, keystrokes, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (exercise_id, student_name, submitted_translation, time_spent, keystrokes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_submissions(exercise_id=None):
    conn = sqlite3.connect("exercises.db")
    cur = conn.cursor()
    if exercise_id:
        cur.execute("SELECT * FROM submissions WHERE exercise_id=?", (exercise_id,))
    else:
        cur.execute("SELECT * FROM submissions")
    rows = cur.fetchall()
    conn.close()
    return rows

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation Training Tool")

init_db()

role = st.sidebar.selectbox("Select Role", ["Instructor", "Student"])

# -----------------------------
# Instructor View
# -----------------------------
if role == "Instructor":
    st.header("👨‍🏫 Instructor Dashboard")

    st.subheader("Create a New Exercise")
    with st.form("create_exercise"):
        title = st.text_input("Exercise Title")
        source_text = st.text_area("Source Text")
        ref_translation = st.text_area("Reference Translation")
        submitted = st.form_submit_button("Add Exercise")

        if submitted:
            if title and source_text:
                add_exercise(title, source_text, ref_translation)
                st.success("✅ Exercise added successfully!")
            else:
                st.error("⚠️ Please provide a title and source text.")

    st.subheader("Available Exercises")
    exercises = get_exercises()
    for ex in exercises:
        st.markdown(f"**{ex[1]}** (ID: {ex[0]})")
        st.text(f"Source: {ex[2]}")
        st.text(f"Reference: {ex[3] if ex[3] else '—'}")

    st.subheader("Student Submissions")
    all_subs = get_submissions()
    if all_subs:
        for sub in all_subs:
            st.markdown(f"📝 Exercise ID: {sub[1]}, Student: {sub[2]}")
            st.text(f"Translation: {sub[3]}")
            st.text(f"Time Spent: {sub[4]:.2f} sec, Keystrokes: {sub[5]}")
            st.caption(f"Submitted at: {sub[6]}")
            st.markdown("---")
    else:
        st.info("No submissions yet.")

# -----------------------------
# Student View
# -----------------------------
if role == "Student":
    st.header("👩‍🎓 Student Dashboard")

    student_name = st.text_input("Enter your name")
    exercises = get_exercises()

    if not exercises:
        st.warning("⚠️ No exercises available yet. Please wait for the instructor.")
    else:
        selected_ex = st.selectbox("Choose an Exercise", exercises, format_func=lambda x: f"{x[1]} (ID {x[0]})")

        if selected_ex:
            st.markdown(f"### Source Text:\n {selected_ex[2]}")

            if student_name:
                start_time = time.time()
                keystrokes = st.session_state.get("keystrokes", 0)

                def count_keystrokes():
                    st.session_state.keystrokes = st.session_state.get("keystrokes", 0) + 1

                student_translation = st.text_area("Your Translation", on_change=count_keystrokes)

                if st.button("Submit Translation"):
                    end_time = time.time()
                    time_spent = end_time - start_time
                    add_submission(selected_ex[0], student_name, student_translation, time_spent, keystrokes)
                    st.success("✅ Translation submitted successfully!")
            else:
                st.error("⚠️ Please enter your name first.")
