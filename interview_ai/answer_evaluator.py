# interview_ai/answer_evaluator.py
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# Load a lightweight model once
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _clean(text: str):
    """Basic normalization for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def evaluate_answer(question: str, candidate_answer: str, reference_answer: str) -> float:
    """
    Evaluate a candidate's answer compared to a reference answer.
    Returns a similarity score between 0.0 and 1.0.
    """

    if not candidate_answer or not reference_answer:
        return 0.0

    # Step 1: lexical similarity (quick surface comparison)
    lex_sim = SequenceMatcher(None, _clean(candidate_answer), _clean(reference_answer)).ratio()

    # Step 2: semantic similarity (deep vector comparison)
    emb_cand = _model.encode(candidate_answer, convert_to_tensor=True)
    emb_ref = _model.encode(reference_answer, convert_to_tensor=True)
    sem_sim = float(util.cos_sim(emb_cand, emb_ref))

    # Weighted score (60% semantic, 40% lexical)
    final_score = round(0.6 * sem_sim + 0.4 * lex_sim, 3)
    return final_score
