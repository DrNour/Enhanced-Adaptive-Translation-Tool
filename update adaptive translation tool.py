import streamlit as st
from bert_score import score

def semantic_eval(candidate, reference):
    try:
        P, R, F1 = score(
            [candidate],
            [reference],
            lang="en",  # can be "ar" if Arabic is target
            model_type="bert-base-multilingual-cased"
        )
        return {
            "BERT_P": round(P.item(), 4),
            "BERT_R": round(R.item(), 4),
            "BERT_F1": round(F1.item(), 4),
        }
    except Exception as e:
        return {"Error": str(e)}
