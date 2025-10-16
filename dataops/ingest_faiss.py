import logging
from pathlib import Path
from collections import Counter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data/raw"
INDEX_DIR = ROOT / "data/indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def ingest_pdfs():
    logging.info("📂 Searching for PDFs in %s", KB_DIR)
    pdf_paths = list(KB_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {KB_DIR}")

    docs = []
    for pdf in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            logging.info("  • Loaded %s (%d pages)", pdf.name, len(pdf_docs))
        except Exception as e:
            logging.warning("  ! Skipped %s (%s)", pdf.name, e)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    logging.info("✂️  Chunked into %d pieces", len(chunks))

    src_counts = Counter(c.metadata.get("source","?") for c in chunks)
    for src, n in src_counts.most_common(10):
        logging.info("  - %s: %d chunks", Path(src).name, n)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info("🧠 Building FAISS index…")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_DIR / "kb_faiss"))
    logging.info("✅ FAISS index saved to %s", INDEX_DIR / "kb_faiss")

if __name__ == "__main__":
    ingest_pdfs()
