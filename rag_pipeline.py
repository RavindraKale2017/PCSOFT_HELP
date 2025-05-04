import fitz  # PyMuPDF
import faiss
import numpy as np
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pickle

PDF_PATH = "Data/Report_Index only.pdf"  # Update this if your folder is 'Data'

def extract_pdf_chunks(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Error opening PDF: {e}")
        return []
    chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks.append(text.strip())
    if not chunks:
        print("[WARNING] No extractable text found in PDF.")
    return chunks

def embed_chunks(chunks):
    print("[INFO] Starting embedding...")
    start = time.time()
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(chunks)
    elapsed = time.time() - start
    print(f"[INFO] Embedding completed in {elapsed:.2f} seconds.")
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    print("[INFO] Building FAISS index...")
    start = time.time()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    elapsed = time.time() - start
    print(f"[INFO] FAISS index built in {elapsed:.2f} seconds.")
    return index

# Extract, embed, and index ONCE at module load
print("[INFO] Extracting PDF chunks...")
INDEX_FILE = "faiss_index.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"

def save_index(index, embeddings):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
        return index, embeddings
    return None, None

index, embeddings = load_index()
chunks = []  # Initialize chunks before conditional check
if index is None:
    chunks = extract_pdf_chunks(PDF_PATH)
    if chunks:
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        save_index(index, embeddings)
else:
    # Load chunks from file if index exists
    chunks = extract_pdf_chunks(PDF_PATH) if not chunks else chunks

# Pre-load the embedder at module level
global_embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve_relevant_chunks(query, top_k=3):
    global chunks, global_embedder
    if not chunks or embeddings is None or index is None:
        return [], []
        
    query_vec = np.array(global_embedder.embed_query(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    retrieved = [chunks[i] for i in I[0]]
    return retrieved, retrieved  # Return exactly 2 values

def save_chunks_to_txt(chunks, out_path="extracted_chunks.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

if __name__ == "__main__":
    if chunks:
        save_chunks_to_txt(chunks)
        print(f"[INFO] Extracted {len(chunks)} chunks from the PDF.")
    else:
        print("[ERROR] No chunks extracted. Please check the PDF path and content.")