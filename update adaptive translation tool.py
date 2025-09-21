def compute_scores(hypothesis, reference):
    """Compute BLEU, chrF, TER, and BERTScore safely."""
    results = {"BLEU": None, "chrF": None, "TER": None, "BERT_F1": None}

    if not hypothesis or not reference:
        st.warning("⚠️ No reference or hypothesis provided. Scores unavailable.")
        return results

    try:
        # BLEU
        results["BLEU"] = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score
    except Exception as e:
        st.error(f"BLEU error: {e}")

    try:
        # chrF
        results["chrF"] = sacrebleu.corpus_chrf([hypothesis], [[reference]]).score
    except Exception as e:
        st.error(f"chrF error: {e}")

    try:
        # TER
        results["TER"] = sacrebleu.corpus_ter([hypothesis], [[reference]]).score
    except Exception as e:
        st.error(f"TER error: {e}")

    try:
        # BERTScore (English only by default)
        P, R, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=True)
        results["BERT_F1"] = float(F1[0])
    except Exception as e:
        st.error(f"BERTScore error: {e}")

    return results
