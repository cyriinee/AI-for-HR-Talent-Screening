import os, re, json, logging
from textwrap import shorten
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

# Load .env for GROQ_API_KEY
load_dotenv()

logging.basicConfig(level=logging.INFO)

ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(ROOT, "data", "indexes")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------
# Load FAISS retriever
# ---------------------------------------------------------------------
def load_retriever(k=4):
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(os.path.join(INDEX_DIR, "kb_faiss"), emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k}), emb


def _trim_context(txt, max_chars=6000):
    return txt[:max_chars]


# ---------------------------------------------------------------------
# Retrieve context from your indexed PDFs
# ---------------------------------------------------------------------
def build_context_for_skills(skills, retriever, k=4):
    query = ", ".join(skills) if isinstance(skills, (list, tuple, set)) else str(skills)
    hits = retriever.invoke(query)[:k]
    ctx = "\n\n".join(
        f"[source: {h.metadata.get('source','?')}, page {h.metadata.get('page','?')}]\n{h.page_content}"
        for h in hits
    )
    return _trim_context(ctx), hits


# ---------------------------------------------------------------------
# Prompt template (Questions + Answers)
# ---------------------------------------------------------------------
def rag_prompt(context, skills, n=5):
    return f"""
You are an interview preparation assistant.

Using ONLY the context below, generate {n} technical interview questions.
For each question, also provide a short, correct answer.

Context:
\"\"\"{context}\"\"\"

Skills to target: {', '.join(skills) if isinstance(skills, (list, tuple, set)) else skills}

Rules:
- Output a valid JSON array.
- Each element must contain:
  {{
    "question": "the interview question",
    "answer": "the correct answer"
  }}
- Keep answers short (1–3 sentences), technically precise, and relevant to the context.
- No markdown or explanations outside the JSON.
"""


# ---------------------------------------------------------------------
# Normalize fallback (if model output isn't valid JSON)
# ---------------------------------------------------------------------
def _normalize_numbered_list(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 1:
        parts = re.split(r"\s*\d+[\)\.]?\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]
    return [re.sub(r"^\s*\d+[\)\.\-\s]*", "", l).strip() for l in lines if l.strip()]


# ---------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------
def generate_questions_for_skills(skills, n=5):
    retriever, _ = load_retriever()
    ctx, hits = build_context_for_skills(skills, retriever)
    prompt = rag_prompt(ctx, skills, n)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_name = "llama-3.3-70b-versatile"

    logging.info("🧠 Generating questions + answers via Groq API...")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    text = resp.choices[0].message.content.strip()
    logging.info("Raw model output:\n%s", text)

    # Try to parse valid JSON
    try:
        qa_pairs = json.loads(text)
        # ensure each item has both question and answer
        if not all("question" in qa and "answer" in qa for qa in qa_pairs):
            raise ValueError("Missing keys in JSON")
    except Exception as e:
        logging.warning("⚠️ Model did not return valid JSON: %s", e)
        # fallback: basic numbered list mode
        qs = _normalize_numbered_list(text)
        qa_pairs = [{"question": q, "answer": ""} for q in qs]

    sources = [
        {"source": h.metadata.get("source", "?"), "page": h.metadata.get("page", "?")}
        for h in hits
    ]

    return {
        "skills": skills,
        "qa_pairs": qa_pairs[:n],
        "sources": sources,
        "context_preview": shorten(ctx, 600)
    }


# ---------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_skills = ["python", "sql", "async javascript"]
    output = generate_questions_for_skills(sample_skills, n=5)
    print(json.dumps(output, indent=2))
