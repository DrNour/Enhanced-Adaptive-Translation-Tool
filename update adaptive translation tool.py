import streamlit as st
import time
import difflib
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import meteor_score
import numpy as np

# ----------------------------
# Simple in-memory storage
# ----------------------------
EXERCISES = {}        # {exercise_id: {"text": ..., "reference": ..., "assigned": [students]}}
SUBMISSIONS = defaultdict(list)  # {student: [{"exercise_id": ..., "submission": ..., "metrics": {...}, "time": ..., "keystrokes": ...}]}

# ----------------------------
# Helpers
# ----------------------------
def calculate_metrics(reference, hypothesis):
    smoothie = SmoothingFunction().method1
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie) if hyp_tokens else 0
    chrf = sentence_chrf([reference], hypothesis) if hypothesis else 0
    meteor = meteor_score([reference], hypothesis) if hypothesis else 0
    edit_distance = np.sum([1 for a, b in zip(reference, hypothesis) if a != b]) + abs(len(reference)-len(hypothesis))
    return {
        "BLEU": round(bleu, 3),
        "chrF": round(chrf, 3),
        "METEOR": round(meteor, 3),
        "Edit Distance": edit_distance
    }

def highlight_diff(reference, student):
    diff = difflib.ndiff(reference.split(), student.split())
    ref_out, stud_out = [], []
    for token in diff:
        if token.startswith("  "):
            ref_out.append(token[2:])
            stud_out.append(token[2:])
        elif token.startswith("- "):
            ref_out.append(f"❌{token[2:]}")
        elif token.startswith("+ "):
            stud_out.append(f"🆕{token[2:]}")
    return " ".join(ref_out), " ".join(stud_out)

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Adaptive Translation Tool", layout="wide")
st.title("🌍 Adaptive Translation Tool")

role = st.selectbox("Select your role", ["Instructor", "Student"])

# ----------------------------
# Instructor Panel
# ----------------------------
if role == "Instructor":
    st.header("👩‍🏫 Instructor Dashboard")

    with st.expander("➕ Create Exercise"):
        ex_id = st.text_input("Exercise ID")
        ex_text = st.text_area("Source Text")
        ex_ref = st.text_area("Reference Translation")
        assign_to = st.text_input("Assign to (comma-separated student names)")
        if st.button("Save Exercise"):
            if ex_id and ex_text:
                EXERCISES[ex_id] = {
                    "text": ex_text,
                    "reference": ex_ref,
                    "assigned": [s.strip() for s in assign_to.split(",")] if assign_to else []
                }
                st.success(f"Exercise '{ex_id}' saved & assigned!")

    st.subheader("📊 Leaderboard")
    scores = []
    for student, subs in SUBMISSIONS.items():
        avg_bleu = np.mean([s["metrics"]["BLEU"] for s in subs])
        scores.append((student, avg_bleu))
    if scores:
        for s, b in sorted(scores, key=lambda x: x[1], reverse=True):
            st.write(f"**{s}** – Avg BLEU: {b:.3f}")

    st.subheader("📂 Student Work")
    for student, subs in SUBMISSIONS.items():
        with st.expander(student):
            for s in subs:
                st.write(f"**Exercise:** {s['exercise_id']}")
                st.write(f"**Submission:** {s['submission']}")
                st.write(f"**Metrics:** {s['metrics']}")
                st.write(f"**Time Spent:** {s['time']} sec")
                st.write(f"**Keystrokes:** {s['keystrokes']}")
                if s['metrics']:
                    ref, stud = highlight_diff(EXERCISES[s['exercise_id']]["reference"], s['submission'])
                    st.markdown(f"**Reference:** {ref}")
                    st.markdown(f"**Student:** {stud}")
                st.write("---")

# ----------------------------
# Student Panel
# ----------------------------
else:
    st.header("🧑‍🎓 Student Dashboard")
    student_name = st.text_input("Enter your name")
    if student_name:
        # Assigned exercises
        assigned = [eid for eid, e in EXERCISES.items() if student_name in e["assigned"]]
        if assigned:
            selected = st.selectbox("Choose an exercise", assigned)
            if selected:
                exercise = EXERCISES[selected]
                st.write(f"**Source Text:** {exercise['text']}")

                # Track editing
                if f"start_time_{selected}" not in st.session_state:
                    st.session_state[f"start_time_{selected}"] = time.time()
                    st.session_state[f"keystrokes_{selected}"] = 0

                submission = st.text_area("Your Translation", key=f"sub_{selected}")
                st.session_state[f"keystrokes_{selected}"] += len(submission) if submission else 0

                if st.button("Submit Translation"):
                    end_time = time.time()
                    elapsed = round(end_time - st.session_state[f"start_time_{selected}"], 2)
                    metrics = calculate_metrics(exercise["reference"], submission) if exercise["reference"] else {}
                    SUBMISSIONS[student_name].append({
                        "exercise_id": selected,
                        "submission": submission,
                        "metrics": metrics,
                        "time": elapsed,
                        "keystrokes": st.session_state[f"keystrokes_{selected}"]
                    })
                    st.success("Submission recorded!")
                    st.write(metrics)
        else:
            st.info("No exercises assigned to you yet.")
