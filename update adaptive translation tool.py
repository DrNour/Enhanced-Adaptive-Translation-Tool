import streamlit as st
import pandas as pd
import nltk
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import sacrebleu

# Download punkt if missing
nltk.download("punkt", quiet=True)

# Try loading BERTScore safely
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

# --- DATA STRUCTURES ---
if "exercises" not in st.session_state:
    st.session_state.exercises = []

if "submissions" not in st.session_state:
    st.session_state.submissions = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "keystrokes" not in st.session_state:
    st.session_state.keystrokes = 0

# --- FUNCTIONS ---
def evaluate_translation(student_translation, reference_translation):
    results = {}

    # BLEU
    try:
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([reference_translation.split()],
                             student_translation.split(),
                             smoothing_function=smoothie)
        results["BLEU"] = round(bleu, 3)
    except:
        results["BLEU"] = None

    # METEOR
    try:
        meteor = meteor_score([reference_translation.split()],
                              student_translation.split())
        results["METEOR"] = round(meteor, 3)
    except:
        results["METEOR"] = None

    # chrF & TER
    try:
        chrf = sacrebleu.corpus_chrf([student_translation],
                                     [[reference_translation]]).score
        ter = sacrebleu.corpus_ter([student_translation],
                                   [[reference_translation]]).score
        results["chrF"] = round(chrf, 3)
        results["TER"] = round(ter, 3)
    except:
        results["chrF"] = None
        results["TER"] = None

    # BERTScore if available
    if BERT_AVAILABLE:
        try:
            P, R, F1 = bert_score([student_translation],
                                  [reference_translation],
                                  lang="en", verbose=False)
            results["BERTScore"] = round(F1.mean().item(), 3)
        except:
            results["BERTScore"] = None
    else:
        results["BERTScore"] = None

    return results

# --- UI ---
st.sidebar.title("Menu")
role = st.sidebar.radio("Choose your role:", ["Instructor", "Student"])

if role == "Instructor":
    st.title("Instructor Dashboard")

    st.subheader("Create a New Exercise")
    exercise_text = st.text_area("Enter source text:")
    reference_translation = st.text_area("Enter reference translation:")
    if st.button("Add Exercise"):
        if exercise_text and reference_translation:
            st.session_state.exercises.append({
                "source": exercise_text,
                "reference": reference_translation
            })
            st.success("Exercise added successfully!")

    st.subheader("All Exercises")
    if st.session_state.exercises:
        for idx, ex in enumerate(st.session_state.exercises, start=1):
            st.write(f"**Exercise {idx}:** {ex['source']}")
            st.write(f"Reference: {ex['reference']}")
    else:
        st.info("No exercises yet.")

    st.subheader("Submissions Summary")
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        st.dataframe(df)
    else:
        st.info("No student submissions yet.")

elif role == "Student":
    st.title("Student Dashboard")

    if not st.session_state.exercises:
        st.warning("No exercises available yet. Please wait for your instructor to add some.")
    else:
        exercise_choice = st.selectbox("Choose an exercise:", range(1, len(st.session_state.exercises)+1))
        chosen_ex = st.session_state.exercises[exercise_choice-1]

        st.write("### Source Text")
        st.info(chosen_ex["source"])

        # Start timer when entering exercise
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        # Student translation input
        student_translation = st.text_area(
            "Enter your translation here:",
            on_change=lambda: st.session_state.update({"keystrokes": st.session_state.keystrokes + 1})
        )

        fluency = st.slider("Rate Fluency (1=Poor, 5=Excellent)", 1, 5, 3)
        semantic = st.slider("Rate Semantic Accuracy (1=Poor, 5=Excellent)", 1, 5, 3)

        if st.button("Submit Translation"):
            if student_translation.strip():
                results = evaluate_translation(student_translation, chosen_ex["reference"])

                # Compute time spent
                time_spent = round(time.time() - st.session_state.start_time, 2)

                # Save submission
                st.session_state.submissions.append({
                    "Exercise": exercise_choice,
                    "Student Translation": student_translation,
                    "BLEU": results["BLEU"],
                    "METEOR": results["METEOR"],
                    "chrF": results["chrF"],
                    "TER": results["TER"],
                    "BERTScore": results["BERTScore"],
                    "Keystrokes": st.session_state.keystrokes,
                    "Time Spent (s)": time_spent,
                    "Fluency (Self)": fluency,
                    "Semantic Accuracy (Self)": semantic
                })

                st.success("Translation submitted and evaluated!")

                st.write("### Your Scores")
                st.json(results)

                st.write(f"**Keystrokes:** {st.session_state.keystrokes}")
                st.write(f"**Time Spent:** {time_spent} seconds")
                st.write(f"**Fluency (self):** {fluency}")
                st.write(f"**Semantic Accuracy (self):** {semantic}")

                # Reset for next attempt
                st.session_state.start_time = None
                st.session_state.keystrokes = 0
            else:
                st.error("Please enter your translation before submitting.")
