from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import logging

# Import from our internal modules
from knowledge_ai.groq_questions import generate_questions_for_skills
from knowledge_ai.skills_plan import compute_skill_plan
from interview_ai.answer_evaluator import evaluate_answer

app = FastAPI(title="GetHired Question Generator API")

# ----------------------------------------------------
# 📘 MODELS
# ----------------------------------------------------
class SkillRequest(BaseModel):
    candidate_skills: List[str]
    job_skills: List[str]

class AnswerRequest(BaseModel):
    questions: List[str]
    candidate_answers: List[str]
    reference_answers: List[str]


# ----------------------------------------------------
# 🧠 QUESTION GENERATION ENDPOINT
# ----------------------------------------------------
@app.post("/generate")
def generate(data: SkillRequest):
    """
    Compare candidate and job skills, find gaps and overlaps,
    then generate technical questions using Groq API.
    """
    logging.info("Generating interview questions...")
    plan = compute_skill_plan(data.candidate_skills, data.job_skills)

    result = {
        "plan": plan,
        "questions": {
            "gaps": generate_questions_for_skills(plan["gaps"], n=5),
            "overlap": generate_questions_for_skills(plan["overlap"], n=5),
        },
    }
    return result


# ----------------------------------------------------
# 🧮 ANSWER EVALUATION ENDPOINT
# ----------------------------------------------------
@app.post("/evaluate")
def evaluate(data: AnswerRequest):
    """
    Evaluate candidate answers against reference answers.
    Returns a similarity score (0 to 1) and average score.
    """
    scores = []
    for q, cand, ref in zip(data.questions, data.candidate_answers, data.reference_answers):
        s = evaluate_answer(q, cand, ref)
        scores.append({
            "question": q,
            "candidate_answer": cand,
            "reference_answer": ref,
            "score": s
        })

    avg = round(sum(s["score"] for s in scores) / len(scores), 3)
    return {"average_score": avg, "results": scores}
