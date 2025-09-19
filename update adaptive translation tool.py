import streamlit as st
import pandas as pd
import difflib
import time

# Optional metrics
try:
    from sacrebleu import sentence_bleu, sentence_chrf, TER
    import evaluate
    comet_metric = evaluate.load("comet")
    bertscore_metric = evaluate.load("bertscore")
except Exception:
    comet_metric = None
    bertscore_metric = None

# -----------------------
# Database Simulation
# -----------------------
if "students" not in st.session_state:
    st.session_state["students"] = {}

if "history" not in st.session_state:
    st.session_state["history"] = []

# -----------------------
# Error Categorization (simple demo)
# -----------------------
def categorize_errors(reference, hypothesis):
    errors = {"Lexical": [], "Grammar": [], "Idiomatic": [], "Semantic": []}
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    
    for w in hyp_words:
        if w not in ref_words:
            if w.endswith("ed") or w.endswith("ing"):
                errors["Grammar"].append(w)
            elif len(w) > 8:
                errors["Lexical"].append(w)
            else:
                errors["Semantic"].append(w)
    if "like" in hypothesis.lower() and "as" not in reference.lower():
        errors["Idiomatic"].append("like/as misuse")
    return errors

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_translation(ref, hyp):
    results = {}
    try:
        results["BLEU"] = sentence_bleu([hyp], [ref]).score
        results["chrF"] = sentence_chrf([hyp], [ref]).score
        results["TER"] = TER().corpus_score([hyp], [ref]).score
    except:
        results["BLEU"] = results["chrF"] = results["TER"] = None
    
    if bertscore_metric:
        try:
            bert_res = bertscore_metric.compute(predictions=[hyp], references=[ref], lang="en")
            results["BERTScore"] = bert_res["f1"][0]
        except:
            results["BERTScore"] = None
    
    if comet_metric:
        try:
            comet_res = comet_metric.compute(predictions=[hyp], references=[ref], sources=[""])
            results["COMET"] = comet_res["mean_score"]
        except:
            results["COMET"] = None
    return results

# -----------------------
# Adaptive Suggestions
# -----------------------
def suggest_exercises(error_summary):
    suggestions = []
    if error_summary["Lexical"]:
        suggestions.append("Revise key vocabulary in context and try synonym exercises.")
    if error_summary["Grammar"]:
        suggestions.append("Practice verb tense drills and sentence structure tasks.")
    if error_summary["Idiomatic"]:
        suggestions.append("Translate idiomatic expressions and proverbs.")
    if error_summary["Semantic"]:
        suggestions.append("Work on precision with collocations and semantic fields.")
    if not suggestions:
        suggestions.append("Excellent! Try a longer and more complex text.")
    return suggestions

# -----------------------
# Streamlit UI
# -----------------------
st.title("EduTransAI – Translation Training Platform")

# Login
student_name = st.text_input("Enter your name to continue:")
if not student_name:
    st.stop()

if student_name not in st.session_state["students"]:
    st.session_state["students"][student_name] = {"points": 0, "submissions": []}

menu = st.sidebar.radio("Menu", ["Dashboard", "Translate", "Leaderboard"])

# Dashboard
if menu == "Dashboard":
    st.subheader(f"Welcome {student_name} 👋")
    data = st.session_state["students"][student_name]
    st.metric("Total Points", data["points"])
    st.write("Your submissions:")
    st.write(pd.DataFrame(data["submissions"]))

# Translate
elif menu == "Translate":
    st.subheader("Translation Exercise")
    source_text = st.text_area("Source Text", "With love’s light wings did I o’er-perch these walls; For stony limits cannot hold love out…")
    reference = st.text_area("Reference Translation", "تسلقت هذه الجدران بجناحي الحب الخفيفين؛ فلا تستطيع حدود الحجارة أن تمنع الحب.")
    
    # Keystroke simulation (timestamp tracking)
    if "keystrokes" not in st.session_state:
        st.session_state["keystrokes"] = []
    
    student_translation = st.text_area("Your Translation")
    if student_translation:
        st.session_state["keystrokes"].append((time.time(), student_translation))
    
    if st.button("Submit Translation"):
        results = evaluate_translation(reference, student_translation)
        errors = categorize_errors(reference, student_translation)
        suggestions = suggest_exercises(errors)

        # Save submission
        st.session_state["students"][student_name]["submissions"].append({
            "Source": source_text,
            "Your Translation": student_translation,
            "Reference": reference,
            "Scores": results,
            "Errors": errors,
            "Suggestions": suggestions,
            "Edits": len(st.session_state["keystrokes"])
        })
        st.session_state["students"][student_name]["points"] += 10
        
        # Display Results
        st.success("Translation submitted and evaluated ✅")
        st.write("### Evaluation Scores", results)
        st.write("### Error Categorization", errors)
        st.write("### Adaptive Suggestions", suggestions)
        st.write("### Edit History (keystrokes tracked)", st.session_state["keystrokes"])

# Leaderboard
elif menu == "Leaderboard":
    st.subheader("Leaderboard 🏆")
    leaderboard = []
    for s, d in st.session_state["students"].items():
        leaderboard.append({"Student": s, "Points": d["points"]})
    df = pd.DataFrame(leaderboard).sort_values("Points", ascending=False)
    st.table(df)
